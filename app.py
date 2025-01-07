import os
import pandas as pd
import streamlit as st
from Levenshtein import ratio
import requests
import time
from io import BytesIO
import concurrent.futures
import re
import Levenshtein

start_time = time.time()

def snapshot_data_fetched(snapshot_id, api_key):
    api_endpoint = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"
    while True:
        response_snap_code = requests.get(api_endpoint, headers={"Authorization": f"Bearer {api_key}"})
        if response_snap_code.status_code == 200:
            snapshot_data = response_snap_code.json()
            return snapshot_data
        else:
            time.sleep(15)


def find_key_recursively(data, target_key):
    results = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                results.append(value)
            elif isinstance(value, (dict, list)):
                results.extend(find_key_recursively(value, target_key))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_key_recursively(item, target_key))
    
    return results
    
def brightdata_api_data_extraction(api_key, api_endpoint, dataset, batch_size=100, delay=15):
    dataset = linkedin_profile_url_validation(dataset)
    linkedin_profiles_urls = dataset['LinkedIn Profile'].tolist()
    linkedin_profiles_organization_name = dataset['Organization Name'].tolist()

    data = []
    total_profiles = len(linkedin_profiles_urls)
    batch_count = (total_profiles + batch_size - 1) // batch_size
    count=0
    for batch_start in range(0, total_profiles, batch_size):
        batch_end = min(batch_start + batch_size, total_profiles)
        batch_urls = linkedin_profiles_urls[batch_start:batch_end]
        batch_orgs = linkedin_profiles_organization_name[batch_start:batch_end]        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_profile = {
                executor.submit(fetch_profile_data, api_key, api_endpoint, url, org): (url, org)
                for url, org in zip(batch_urls, batch_orgs)
            }
            
            batch_data = []
            for future in as_completed(future_to_profile):
                result = future.result()
                batch_data.append(result)
                count=count+1

        data.extend(batch_data)
        if batch_start + batch_size < total_profiles:
            time.sleep(delay)
    
    brightdata_api_extracted_data = pd.DataFrame(data)
    return brightdata_api_extracted_data

# LinkedIn Profile Validation
def linkedin_profile_url_validation(dataset):
    linkedin_regex = r'^https:\/\/(www\.)?linkedin\.com\/in\/[a-zA-Z0-9\-%]+\/?$'
    validated_dataset = dataset[dataset['LinkedIn Profile'].apply(lambda url: bool(re.match(linkedin_regex, url))) ]
    return validated_dataset

def is_similar_company(company1, company2, threshold=60):
    similarity_score = ratio(company1.lower(), company2.lower()) * 100
    return similarity_score >= threshold

def process_experience_data(current_company, experiences, platform_company):
    """
    Process LinkedIn experience data to check if someone has moved to another organization.
    Args:
        experiences (list): List of experience dictionaries.
        platform_company (str): The company name from your platform.

    Returns:
        dict: Information about whether the person moved organizations or not.
    """
    matches = []
    for exp in experiences:
        company = exp.get("subtitle") or exp.get("company")
        end_date = find_key_recursively(exp, "end_date")
        
        if company and "Present" in end_date:
            similarity = ratio(company.lower(), platform_company.lower()) * 100
            matches.append({"company": company, "similarity": similarity, "end_date": end_date})    
    
    if len(matches)>0:
        maximum_score_company=max(matches, key=lambda x: x['similarity'])['company']
        if is_similar_company(maximum_score_company, platform_company):
            moved = False
        else:
            moved = True
    else:
        if is_similar_company(current_company, platform_company):
            moved = False
        else:
            moved = True
        
    return {
        "matches": matches,
        "moved": moved,
    }
    def fetch_profile_data(api_key, api_endpoint, url, org):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {"url": url}
    while True:
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response_json = response.json()
            snapshot_id = response_json.get("snapshot_id")
            snapshot_data = snapshot_data_fetched(snapshot_id, api_key)
            break
        except:
            pass
        
    if not snapshot_data:
        return {
            "Previous Organization Name": org,
            "LinkedIn Profile URL": url,
            "Current Organization Name": "Not Found"
        }
    try:
        current_company = snapshot_data[0]['current_company']['name']
        experiences = snapshot_data[0].get("experience", [])
        results = process_experience_data(current_company, experiences, org)
        status = "Yes" if results["moved"] else "No"  
        return {
            "Previous Organization Name": org,
            "LinkedIn Profile URL": url,
            "Current Organization Name": current_company,
            "Job Change Status": status
        }
        
    except Exception as e:
        return {
            "Previous Organization Name": org,
            "LinkedIn Profile URL": url,
            "Current Organization Name": "Profile Access Restriction",
            "Error": str(e)
        }


st.title("JobChange - Tracker")
uploaded_file = st.file_uploader(
    "Upload input file (Excel format only) - Required Columns(LinkedIn Profile & Organization Name):", type=["xlsx"]
)

# Initialize session state to store generated files
if "original_file" not in st.session_state:
    st.session_state.original_file = None
if "updated_file" not in st.session_state:
    st.session_state.updated_file = None
    
if uploaded_file is not None:
    st.write("Uploaded file:", uploaded_file.name)
    dataset = pd.read_excel(uploaded_file)
    #api_key = '952aa6b01ab4872ef9e8f731c517bea64540b6d132052ff919a33bfe7b703f58' # old paid account API
    api_key = 'd6bcc86c8302c2d740f5c1b3d002e6ae39940c2e2d5cb0fc48267afb18e77425' # new testing account API
    api_endpoint = r'https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_l1viktl72bvl7bjuj0&include_errors=true'
    
    if st.button("Tracker"):
        if "LinkedIn Profile" in dataset.columns and "Organization Name" in dataset.columns:
            try:
                st.info("Tracking PeopleMovement... Please wait.")
                brightdata_api_data = brightdata_api_data_extraction(api_key, api_endpoint, dataset)
                
                if not brightdata_api_data.empty:
                    filtered_brightdata_api_data = brightdata_api_data[brightdata_api_data['Current Organization Name']!="Profile Access Restriction"]
                    filtered_brightdata_api_data = filtered_brightdata_api_data[filtered_brightdata_api_data['Current Organization Name']!="Not Found"]
                    jobchangedprofiles = filtered_brightdata_api_data[filtered_brightdata_api_data['Job Change Status'] == 'Yes']
                    jobchangedprofiles = jobchangedprofiles.drop(columns='Error', errors='ignore')

                    #jobchangedprofiles.drop(columns='Job Change Status', inplace=True)
                    
                    # Logic to update organization names in original dataset
                    dataset['Organization Name'] = dataset['Organization Name'].apply(
                        lambda x: jobchangedprofiles.loc[jobchangedprofiles['Previous Organization Name'] == x, 'Current Organization Name'].values[0] 
                        if x in jobchangedprofiles['Previous Organization Name'].values else x)

                    # Save both original and updated files
                    current_date = pd.Timestamp.today().date()
                    #original_file_path = f"Original_Historical_Data_{current_date}.xlsx"
                    #updated_file_path = f"JobChangesProfiles_{current_date}.xlsx"
                    
                    original_file = BytesIO()
                    updated_file = BytesIO()
                
                    dataset.to_excel(original_file, index=False, sheet_name="Original Data")
                    jobchangedprofiles.to_excel(updated_file, index=False, sheet_name="Updated Data")

                    original_file.seek(0)
                    updated_file.seek(0)
                    
                    st.session_state.original_file = original_file
                    st.session_state.updated_file = updated_file
                    st.success("Files are ready for download!")
                    
                    #with pd.ExcelWriter(original_file_path, engine='openpyxl') as writer:
                     #   dataset.to_excel(writer, sheet_name='HistoricalData', index=False)

                    #with pd.ExcelWriter(updated_file_path, engine='openpyxl') as writer:
                     #   jobchangedprofiles.to_excel(writer, sheet_name='JobChanges', index=False)
                    
                    #st.success(f"Files generated successfully: {original_file_path}, {updated_file_path}")
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    seconds = total_time % 60
                    st.success(f"Total Execuation Time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

                else:
                    st.warning("No significant changes detected.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("The file must contain 'LinkedIn Profile' and 'Organization Name' columns.")

if st.session_state.original_file is not None and st.session_state.updated_file is not None:
    st.download_button(
        label="Download Updated Profiles with Current Organization Names",
        data=st.session_state.original_file,
        file_name="Updated_Historical_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        label="Download Profiles of People Who Changed Jobs",
        data=st.session_state.updated_file,
        file_name="JobChangesProfiles.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
