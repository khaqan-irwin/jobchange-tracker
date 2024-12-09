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
    # Calculate similarity ratio (0-100 scale)
    similarity_score = Levenshtein.ratio(org_name1.lower(), org_name2.lower()) * 100
    # Check if the similarity score meets the threshold
    return similarity_score >= threshold

def fetch_profile_data(api_key, api_endpoint, url, org):
	# Set up headers for the API request
	headers = {
	"Authorization": f"Bearer {api_key}",
	"Content-Type": "application/json"
	}

	# Prepare the payload for the API call
	payload = {"url": url}

	# Make a POST request to fetch profile data
	response = requests.post(api_endpoint, headers=headers, json=payload)
	response_json = response.json()

	# Extract snapshot ID from the response
	snapshot_id = response_json.get("snapshot_id")

	# Fetch additional snapshot data using a helper function
	snapshot_data = snapshot_data_fetched(snapshot_id, api_key)
	if not snapshot_data:
		return {
				"Previous Organization Name": org,
				"LinkedIn Profile URL": url,
				"Current Organization Name": "Not Found"
		}
	try:
		# Extract current company information from the snapshot data
		current_company = snapshot_data[0]['current_company']['name']

		# Extract the list of experiences (jobs) from the snapshot data
		experiences = snapshot_data[0].get("experience", [])

		# Step 1: Identify similar companies in the user's experience
		matches = []  # List to store companies with similarity >= 42

		for exp in experiences:
			# Check if 'company' exists and is the current job ('end_date' == 'Present')
			if exp.get("company") and exp.get("end_date") == "Present":
				# Calculate similarity ratio between the company and the given organization
				similarity_score = Levenshtein.ratio(exp["company"].lower(), org.lower()) * 100
				# If similarity score is above the threshold (42), add to the matches list
				if similarity_score >= 42:
					matches.append((exp["company"], similarity_score))

		# Step 2: Process matches to find the highest similarity company
		if matches:
			# Find the company with the highest similarity score
			max_match = max(matches, key=lambda x: x[1])  # Find the tuple with the highest score
			selected_company = max_match[0]  # Extract the company name
			max_similarity_score = max_match[1]  # Extract the similarity score

			# Step 3: Determine the job change status
			if is_similar(selected_company, org, threshold=max_similarity_score):
				# If the selected company is highly similar to the given organization
				job_status = "Still in previous organization"
			else:
				# If the selected company is not similar enough
				job_status = "Moved from previous organization"
		else:
			# If no matches are found, assume the user moved from the previous organization
			selected_company = None
			max_similarity_score = None
			job_status = "Moved from previous organization"
		# Step 4: Return the results as a dictionary
		return {
			"Previous Organization Name": org,
			"LinkedIn Profile URL": url,
			"Current Organization Name": current_company,
			"Job Change Status": job_status
		}

	# Handle any errors or exceptions that occur
	except Exception as e:
		# Return a response indicating restricted profile access or an error
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
                    jobchangedprofiles = filtered_brightdata_api_data[filtered_brightdata_api_data['Job Change Status'] == 'Moved from previous organization']
                    jobchangedprofiles.drop(columns='Job Change Status', inplace=True)
                    
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
        label="Download Original Dataset",
        data=st.session_state.original_file,
        file_name="Original_Historical_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        label="Download Updated Dataset",
        data=st.session_state.updated_file,
        file_name="JobChangesProfiles.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )