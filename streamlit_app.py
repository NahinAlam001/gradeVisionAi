import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pickle
import streamlit.components.v1 as components

# Load the model from the .pkl file
with open('./model/rf_model_weights.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Set background image
background_image = './path/to/your/background/image.jpg'
components.html(
    f"""
    <style>
        body {{
            background-image: url("{background_image}");
            background-size: cover;
        }}
    </style>
    """,
    height=1,
)

# Display Title and Description with styling
st.title("GradeVisionAI: Student Performance Prediction")
st.markdown(
    """
    <style>
        .big-font {
            font-size: 2.5rem !important;
        }
        .italic {
            font-style: italic;
        }
        .highlight {
            background-color: #FFFF00;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="big-font">Answer the questions below:</p>',
    unsafe_allow_html=True,
)

# Establishing the Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Fetch existing data
existing_data = conn.read(worksheet="Students", usecols=list(range(13)), ttl=1300)
existing_data = existing_data.dropna(how="all")

# List of Business Types and Products
GENDER = [
    'Female', 'Male'
]

UNIVERSITY = [
    'NSU', 'IUBAT', 'BRAC University', 'IUB', 'University of Dhaka',
    'ULAB', 'UITS', 'IUT', 'DIU', 'BUP', 'AIUB',
    'City Medical College', 'AUST',
    'Jashore University of Science and Technology',
    'Dhaladia Degree College', 'University of Asia Pacific', 'EWU',
    'Dhaka Medical College', 'BGMEA', 'University of Rajshahi',
    'Shanto-Mariam University of Creative Technology', 'CUET, DU, HEU',
    'Sylhet Agricultural University', 'National University',
    'Netrokona Medical College', 'Eden Mohila College',
    'Gazipur Govt Mohila College', 'Mymensingh Engineering College',
    'Southeast University', 'UIU', 'MIST', 'Uttara University', 'BUET',
    'Bhawal Badre Alom Govt. College', 'University of Barishal',
    'Bangabandhu Sheikh Mujibur Rahman Agriculture University',
    'Model Institute of Science and Technology',
    'Dream University of Bangladesh',
    'National College of Home Economics',
    'Govt College of Applied Human Science', 'SUST',
    'Comilla Medical College', 'MBSTU',
    'Bangladesh Home Economics College', 'Jagannath University',
    'Northern University Bangladesh',
    'Bangabandhu Sheikh Mujibur Rahman Aviation and Aerospace University',
    'Mawlana Bhashani Science and Technology University', 'JU'
]

YEAR = [
    '1st', '2nd', '3rd', '4th', 'Post-Graduate or Others'
]

MAJOR = [
    'CSE', 'English', 'Law', 'Biochemistry & Biotechnology', 'CS',
    'Biochemistry & Molecular Biology', 'English & Humanities', 'EEE',
    'Marketing', 'Finance', 'UNKNOWN', 'HRM', 'HRM & Marketing',
    'Finance & Banking', 'MIS', 'Biotechnology', 'Literature', 'LLB',
    'Management', 'Environmental Science', 'Finance & Economics',
    'Medical', 'Microbiology', 'Yarn', 'CEE',
    'Environmental Science and Technology', 'Sociology', 'Pharmacy',
    'IPE', 'Bangla', 'CE', 'Applied Statistics', 'Public Health',
    'Accounting', 'Designing', 'Apparel', 'Economics', 'Finance & MIS',
    'Biochemistry', 'EEE, NE', 'Agricultural Engineering & Technology',
    'Botany', 'Psychology', 'Civil Engineering', 'IR', 'CSE.',
    'Philosophy', 'Nuclear Science and Engineering', 'Fine Arts',
    'Geology and Mining', 'International Business',
    'Population Science and Human Resources Development',
    'Material Science', 'Political Science', 'Fisheries',
    'Apparel Manufacturing & Technology',
    'Soil and Environmental Sciences', 'Art and Creative Studies',
    'Textile', 'Resources Management and Entrepreneurship',
    'Child Development and Social Relationship', 'Resource Management',
    'Information Science and Library Management',
    'Immunology, Genetic Engineering,Clinical Biochemistry',
    'Merchandising', 'Fashion Designing & Technology',
    'Food and Nutrition', 'Structure', 'Physics', 'Finance and MIS',
    'Media Communication and Journalism', 'Architecture',
    'Epidemiology', 'Mechanical Engineering',
    'Supply Chain Management',
    'Materials and Metallurgical Engineering',
    'Resource Management and Entrepreneurship',
    'Aerospace Engineering', 'Civil', 'ECE',
    'Marketing & International Business', 'Marketing & MIS'
]

ATTENDANCE = [
    '75-100', '50-75', '25-50', '10-25', '0-10'
]

# Onboarding New Student Form
with st.form(key="student_form"):
    gender = st.selectbox("Gender*", options=GENDER, index=None)
    university = st.selectbox("University*", options=UNIVERSITY, index=None)
    year = st.selectbox("Year*", options=YEAR, index=None)
    major = st.selectbox("Major*", options=MAJOR, index=None)
    interest = st.slider("Interest felt in Major*", 0, 5, 1)
    support = st.slider("Support From Parents*", 0, 5, 1)
    visit_resource = st.slider("How much do you visit study resources?*", 0, 5, 1)
    announcement_check = st.slider("How often do you check your announcements?*", 0, 5, 1)
    participation = st.slider("How much do you participate in class*", 0, 5, 1)
    attendance = st.selectbox("Attendence in percentage*", options=ATTENDANCE, index=None)
    study_hours = st.slider("Study Hours Per Day*", 0, 24, 1)
    group_study = st.slider("Participation in Group Study*", 0, 5, 1)
    stress_level = st.slider("Current Stress Level*", 0, 5, 1)

    # Mark mandatory fields
    st.markdown("**required*")

    submit_button = st.form_submit_button(label="Submit Student Details")

    # If the submit button is pressed
    if submit_button:
        # Create a new row of student data
        student_data = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "University": university,
                    "Year": year,
                    "Major": major,
                    "Interest": interest,
                    "Support": support,
                    "Visit Resource": visit_resource,
                    "Announcement Check": announcement_check,
                    "Class Participation": participation,
                    "Attendance": attendance,
                    "Study Hours": study_hours,
                    "Group Study": group_study,
                    "Stress Level": stress_level
                }
            ]
        )

        # Label Encoding for nominal variables
        le = LabelEncoder()
        nominal_features = ['Gender', 'University', 'Major']
        for feature in nominal_features:
            student_data[feature] = le.fit_transform(student_data[feature])

        # Ordinal Encoding for ordinal variables
        oe = OrdinalEncoder(categories=[['1st', '2nd', '3rd', '4th', 'Post-Graduate or Others'], ['10-25', '25-50', '0-10', '<50', '50-75', '75-100']])
        ordinal_features = ['Year', 'Attendance']
        student_data[ordinal_features] = oe.fit_transform(student_data[ordinal_features])

        # Convert '<1' in 'Study Hours' to 0
        student_data['Study Hours'] = student_data['Study Hours'].replace('<1', 0).astype(int)

        # Apply the same one-hot encoding used during training
        student_data_encoded = pd.get_dummies(student_data, columns=['Gender', 'University', 'Year', 'Major', 'Attendance'])

        # Make sure the columns in the new data match the columns used during training
        missing_cols = set(existing_data.columns) - set(student_data_encoded.columns)
        extra_cols = set(student_data_encoded.columns) - set(existing_data.columns)

        # Add missing columns
        for col in missing_cols:
            student_data_encoded[col] = 0  # or another default value

        # Remove extra columns
        for col in extra_cols:
            student_data_encoded = student_data_encoded.drop(col, axis=1)

        # Ensure that the training data includes the 'CGPA' feature
        if 'CGPA' not in existing_data.columns:
            existing_data['CGPA'] = 0  # or another default value

        # Ensure that the student data includes the 'Stress Level' feature
        if 'Stress Level' not in student_data.columns:
            student_data['Stress Level'] = 0  # or another default value

        # Drop 'CGPA' column if present
        existing_data = existing_data.drop('CGPA', axis=1, errors='ignore')

        # Define the desired order of columns
        desired_columns_order = [
            'Interest', 'Support', 'Visit Resource', 'Announcement Check', 'Class Participation',
            'Study Hours', 'Group Study', 'Stress Level', 'Year', 'University', 'Gender',
            'Attendance', 'Major'
        ]

        # Reorder columns in both DataFrames
        existing_data = existing_data[desired_columns_order]
        student_data_encoded = student_data_encoded[desired_columns_order]

        # Ensure that both DataFrames have the same columns
        missing_cols_student = set(existing_data.columns) - set(student_data_encoded.columns)
        missing_cols_existing = set(student_data_encoded.columns) - set(existing_data.columns)

        # Add missing columns to student_data_encoded
        for col in missing_cols_student:
            student_data_encoded[col] = 0  # or another default value

        # Ensure 'Major' is included in existing_data
        if 'Major' not in existing_data.columns:
            existing_data['Major'] = 0  # or another default value

        # Remove extra columns from student_data_encoded
        for col in missing_cols_existing:
            if col != 'Major':  # Ensure 'Major' is not removed
                student_data_encoded = student_data_encoded.drop(col, axis=1)

        # Ensure 'Major' is included in both DataFrames
        if 'Major' not in existing_data.columns:
            st.error("Error: 'Major' column is missing in existing_data.")
        if 'Major' not in student_data_encoded.columns:
            st.error("Error: 'Major' column is missing in student_data_encoded.")

        # Now, both DataFrames should have the same columns and 'Major' included
        common_columns = set(student_data_encoded.columns) & set(existing_data.columns)

        # Extract the features used during training
        training_features = existing_data.columns[:-1].to_list()  # Assuming the last column is the target variable ('Class')

        # Ensure that 'Major' is included in the training features
        if 'Major' not in training_features:
            training_features.append('Major')

        # Make sure the feature names match the order in student_data_encoded
        training_features_order = [
            'Interest', 'Support', 'Visit Resource', 'Announcement Check', 'Class Participation',
            'Study Hours', 'Group Study', 'Stress Level', 'Year', 'University', 'Gender',
            'Attendance', 'Major'
        ]

        # Reorder columns in training_features
        training_features = [feature for feature in training_features_order if feature in training_features]

        # Make prediction using the loaded model
        predicted_class = rf_model.predict(student_data_encoded[training_features].values.reshape(1, -1))[0]

        # Map predicted class to performance level
        performance_level = "High Performance Expected" if predicted_class == 1 else "Low Performance Expected"

        # Add the predicted performance level to the student data
        student_data_encoded["Performance Level"] = performance_level

        # Select only the relevant features for the model from student_data_encoded
        input_features = student_data_encoded[training_features + ["Performance Level"]]

        # Ensure that the input features have the same order as the training features
        input_features = input_features[training_features_order + ["Performance Level"]]

        # Add the new student data to the existing data
        updated_df = pd.concat([existing_data, input_features], ignore_index=True)

        # Update Google Sheets with the new student data
        conn.update(worksheet="Students", data=updated_df)

        # Display animated success message with a badge for "High Performance Expected"
        if predicted_class == 1:
            st.balloons()
            st.success("High Performance Expected!")

            # You can add more styling to the success message if desired
            st.markdown(
                """
                <style>
                    .highlight-success {
                        color: green;
                        font-size: 1.5rem;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Display predicted class and performance level in the success message
            st.markdown(
                f'<p class="highlight-success">Student details successfully submitted! Predicted Class: {predicted_class}, Performance Level: {performance_level}</p>',
                unsafe_allow_html=True,
            )

        # Display animated error message for "Low Performance Expected"
        else:
            st.error("Low Performance Expected!")

            # You can add more styling to the error message if desired
            st.markdown(
                """
                <style>
                    .highlight-error {
                        color: red;
                        font-size: 1.5rem;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Display predicted class and performance level in the error message
            st.markdown(
                f'<p class="highlight-error">Student details successfully submitted! Predicted Class: {predicted_class}, Performance Level: {performance_level}</p>',
                unsafe_allow_html=True,
            )
