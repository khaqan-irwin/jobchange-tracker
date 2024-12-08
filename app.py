Python 3.12.7 (tags/v3.12.7:0b05ead, Oct  1 2024, 03:06:41) [MSC v.1941 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
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

# Function to fetch profile data

def fetch_profile_data(api_key, api_endpoint, url, org):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"url": url}
    response = requests.post(api_endpoint, headers=headers, json=payload)
    response_json = response.json()

    data = {}
    snapshot_id = response_json.get("snapshot_id")
    snapshot_data = snapshot_data_fetched(snapshot_id, api_key)

    if len(snapshot_data) == 1:
        try:
            if 'current_company' in snapshot_data[0]:
                company_name = snapshot_data[0]['current_company']['name']
                data = {
                    "Previous Organization Name": org,
                    "LinkedIn Profile URL": url,
                    "Current Organization Name": company_name
                }
            else:
                data = {
                    "Previous Organization Name": org,
                    "LinkedIn Profile URL": url,
                    "Current Organization Name": "Profile Access Restriction"
                }
        except Exception as e:
            data = {
                "Previous Organization Name": org,
                "LinkedIn Profile URL": url,
                "Current Organization Name": "Profile Access Restriction"
            }
    else:
        data = {
            "Previous Organization Name": org,
            "LinkedIn Profile URL": url,
            "Current Organization Name": 'Not Found'
        }

    return data


# Parallel processing with ThreadPoolExecutor
def brightdata_api_data_extraction(api_key, api_endpoint, dataset):
    dataset = linkedin_profile_url_validation(dataset)
    linkedin_profiles_urls = dataset['LinkedIn Profile'].tolist()[700:730]
    linkedin_profiles_organization_name = dataset['Organization Name'].tolist()[700:730]
    
    data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(fetch_profile_data, api_key, api_endpoint, url, org)
            for url, org in zip(linkedin_profiles_urls, linkedin_profiles_organization_name)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                data.append(result)
            except Exception as e:
                st.error(f"An error occurred during parallel processing: {e}")

    brightdata_api_extracted_data = pd.DataFrame(data)
    return brightdata_api_extracted_data

# LinkedIn Profile Validation
def linkedin_profile_url_validation(dataset):
    linkedin_regex = r'^https:\/\/(www\.)?linkedin\.com\/in\/[a-zA-Z0-9\-%]+\/?$'
    validated_dataset = dataset[dataset['LinkedIn Profile'].apply(lambda url: bool(re.match(linkedin_regex, url))) ]
    return validated_dataset

# Add similarity scores using Levenshtein ratio
def add_levenshtein_scores(dataframe):
    similarity_scores = []
    for _, row in dataframe.iterrows():
        current_org = row['Current Organization Name']
        previous_org = row['Previous Organization Name']

        if pd.isna(current_org) or pd.isna(previous_org):
            similarity_scores.append(None)
        else:
            score = round(ratio(current_org.lower().strip(), previous_org.lower().strip()) * 100, 2)
            similarity_scores.append(score)
    dataframe['Similarity Score'] = similarity_scores
    return dataframe

def is_similar(org_name1, org_name2, threshold=42):
    similarity_ratio = Levenshtein.ratio(org_name1.lower(), org_name2.lower()) * 100
    return similarity_ratio >= threshold


# Streamlit App UI
#st.title("JobChange - Tracker")
#uploaded_file = st.file_uploader("Upload input file (Excel format only) - Required Columns(LinkedIn Profile & Organization Name):", type=["xlsx"])

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
    api_key = '6a60676872c721aa7ebf3d40100f5801db9c3dabc3d49d77b170e3174c4e3624'
    api_endpoint = r'https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_l1viktl72bvl7bjuj0&include_errors=true'
    
    if st.button("Tracker"):
        if "LinkedIn Profile" in dataset.columns and "Organization Name" in dataset.columns:
            try:
                st.info("Tracking PeopleMovement... Please wait.")
                brightdata_api_data = brightdata_api_data_extraction(api_key, api_endpoint, dataset)
                
                if not brightdata_api_data.empty:
                    filtered_brightdata_api_data = brightdata_api_data[brightdata_api_data['Current Organization Name']!="Profile Access Restriction"]
                    filtered_brightdata_api_data = filtered_brightdata_api_data[filtered_brightdata_api_data['Current Organization Name']!="Not Found"]
                    jobchangedprofiles_full = add_levenshtein_scores(filtered_brightdata_api_data)
                    jobchangedprofiles = jobchangedprofiles_full[jobchangedprofiles_full['Similarity Score'] < 42]
                    jobchangedprofiles.drop(columns='Similarity Score', inplace=True)
                    #jobchangedprofiles['Job Change Status'] = 'Moved from previous organization'
                    
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
...                      #   jobchangedprofiles.to_excel(writer, sheet_name='JobChanges', index=False)
...                     
...                     #st.success(f"Files generated successfully: {original_file_path}, {updated_file_path}")
...                     
...                     end_time = time.time()
...                     total_time = end_time - start_time
...                     hours = int(total_time // 3600)
...                     minutes = int((total_time % 3600) // 60)
...                     seconds = total_time % 60
...                     st.success(f"Total Execuation Time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
... 
...                 else:
...                     st.warning("No significant changes detected.")
...             except Exception as e:
...                 st.error(f"An error occurred: {e}")
...         else:
...             st.error("The file must contain 'LinkedIn Profile' and 'Organization Name' columns.")
... 
... if st.session_state.original_file is not None and st.session_state.updated_file is not None:
...     st.download_button(
...         label="Download Original Dataset",
...         data=st.session_state.original_file,
...         file_name="Original_Historical_Data.xlsx",
...         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
...     )
... 
...     st.download_button(
...         label="Download Updated Dataset",
...         data=st.session_state.updated_file,
...         file_name="JobChangesProfiles.xlsx",
...         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
