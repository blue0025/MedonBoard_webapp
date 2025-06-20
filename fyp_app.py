import streamlit as st
import pandas as pd
import pickle
import os
import spacy

# ========== PATH SETUP ==========
NER_MODEL_PATH = os.path.join("ner_model")
CLASSIFIED_DATA = os.path.join("classified_data.csv")
DISEASE_DB = os.path.join("disease_knowledge.csv")
MEDICINE_DB = os.path.join("medicine_data.csv")

# ========== LOAD MODELS ==========
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

nlp = spacy.load(NER_MODEL_PATH)

# ========== USERS ==========
users = {
    "expert": {"password": "admin123", "role": "expert"},
    "trainee": {"password": "view123", "role": "trainee"},
}

# ========== SESSION STATE ==========
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

# ========== LOGIN PAGE ==========
if not st.session_state.logged_in:
    st.title("🔐 MedOnBoard Login")
    st.subheader("Empowering Medical Learning")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    selected_role = st.selectbox("Login As", ["Select Role", "expert", "trainee"])

    if st.button("Login"):
        user = users.get(username)
        if user and user["password"] == password and user["role"] == selected_role:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
            st.rerun()
        else:
            st.error("Invalid credentials or role.")

# ========== MAIN APP ==========
else:
    st.sidebar.success(f"Logged in as: {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ========== PAGE NAVIGATION ==========
    if st.session_state.role == "expert":
        pages = ["Home", "Disease", "Medicine", "Case Study", "Add Case Study"]
    else:
        pages = ["Home", "Disease", "Medicine", "Case Study"]

    selection = st.sidebar.radio("Navigate", pages)

    # ========== HOME ==========
    if selection == "Home":
        st.title("🏥 MedOnBoard")
        st.markdown("A platform for exploring and understanding real-world medical case data.")
        st.markdown("Use the sidebar to explore Diseases, Medicines, and Case Studies.")

    # ========== DISEASE ==========
    elif selection == "Disease":
        st.title("🦠 Disease Explorer")
        if os.path.exists(DISEASE_DB):
            df = pd.read_csv(DISEASE_DB)
            disease_names = df["Name"].unique().tolist()
            selected = st.selectbox("Choose a Disease", disease_names)
            disease_row = df[df["Name"] == selected].iloc[0]

            st.subheader("📌 Description & Symptoms")
            st.write(f"**Symptoms:** {disease_row['Symptoms']}")
            st.write(f"**Treatments:** {disease_row['Treatments']}")

            if os.path.exists(CLASSIFIED_DATA):
                case_df = pd.read_csv(CLASSIFIED_DATA)
                related_cases = case_df[
                    (case_df["category"] == "Case Study") &
                    (case_df["text"].str.contains(selected, case=False, na=False))
                ]
                st.markdown("### 📚 Related Case Studies")
                if not related_cases.empty:
                    for _, row in related_cases.iterrows():
                        st.markdown(f"- {row['text'][:150]}...")
                else:
                    st.info("No related case studies.")
        else:
            st.warning("Disease knowledgebase not available.")

    # ========== MEDICINE ==========
    elif selection == "Medicine":
        st.title("💊 Medicine Reference")
        if os.path.exists(MEDICINE_DB):
            df = pd.read_csv(MEDICINE_DB)
            medicine_names = df["name"].unique().tolist()
            selected = st.selectbox("Choose a Medicine", medicine_names)
            med_row = df[df["name"] == selected].iloc[0]

            st.subheader("📋 Details")
            st.write(f"**Description:** {med_row['description']}")
            st.write(f"**Indication:** {med_row['indication']}")
            st.write(f"**Dosage:** {med_row['dosage']}")

            if os.path.exists(CLASSIFIED_DATA):
                case_df = pd.read_csv(CLASSIFIED_DATA)
                related_cases = case_df[
                    (case_df["category"] == "Case Study") &
                    (case_df["text"].str.contains(selected, case=False, na=False))
                ]
                st.markdown("### 📚 Case Studies Using This Medicine")
                if not related_cases.empty:
                    for _, row in related_cases.iterrows():
                        st.markdown(f"- {row['text'][:150]}...")
                else:
                    st.info("No related case studies.")
        else:
            st.warning("Medicine knowledgebase not found.")

    # ========== CASE STUDY ==========
    elif selection == "Case Study":
        st.title("📚 Case Study Viewer")
        if os.path.exists(CLASSIFIED_DATA):
            df = pd.read_csv("classified_data.csv")
            case_df = df[df["category"] == "Case Study"]

            if not case_df.empty:
                st.dataframe(case_df)
            
                st.download_button(
                    label="⬇️ Download Case Studies as CSV",
                    data=case_df.to_csv(index=False).encode("utf-8"),
                    file_name="case_studies.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No case studies available yet.")

       # -------------------------------
# ➕ ADD CASE STUDY (EXPERT ONLY)
# -------------------------------
elif selection == "Add Case Study" and st.session_state.role == "expert":
    st.title("📝 Add & Classify a New Medical Note")
    new_case_text = st.text_area("Enter full medical case note", key="expert_input")

    if st.button("Analyze and Review Entities", key="analyze_button"):
        if new_case_text.strip():
            input_vec = vectorizer.transform([new_case_text])
            predicted_category = model.predict(input_vec)[0]

            st.success(f"Predicted Category: **{predicted_category}**")

            # Run NER
            doc = nlp(new_case_text)
            diseases = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
            symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
            medicines = [ent.text for ent in doc.ents if ent.label_ == "MEDICINE"]

            # Editable review fields
            disease_str = st.text_input("🔬 DISEASE Entities", ", ".join(diseases))
            symptom_str = st.text_input("🤒 SYMPTOM Entities", ", ".join(symptoms))
            medicine_str = st.text_input("💊 MEDICINE Entities", ", ".join(medicines))

            # Optional override category
            final_category = st.selectbox("Select Final Category", ["Case Study", "Disease", "Medicine"], index=0)

            # Optional feedback
            dev_feedback = st.text_area("🛠 Developer Feedback (optional)", placeholder="Mention any NER errors or suggestions...")

            if st.button("✅ Save Entry", key="final_save"):
                new_records = []

                # Save full text as main entry
                main_record = {
                    "text": new_case_text,
                    "category": final_category,
                    "diseases": disease_str,
                    "symptoms": symptom_str,
                    "medicines": medicine_str,
                    "feedback": dev_feedback,
                    "author": st.session_state.username,
                    "timestamp": pd.Timestamp.now()
                }
                new_records.append(main_record)

                # Add sub-entries if applicable
                for disease in disease_str.split(","):
                    disease = disease.strip()
                    if disease:
                        new_records.append({
                            "text": disease,
                            "category": "Disease",
                            "author": st.session_state.username,
                            "timestamp": pd.Timestamp.now()
                        })

                for symptom in symptom_str.split(","):
                    symptom = symptom.strip()
                    if symptom:
                        new_records.append({
                            "text": symptom,
                            "category": "Symptom",
                            "author": st.session_state.username,
                            "timestamp": pd.Timestamp.now()
                        })

                for med in medicine_str.split(","):
                    med = med.strip()
                    if med:
                        new_records.append({
                            "text": med,
                            "category": "Medicine",
                            "author": st.session_state.username,
                            "timestamp": pd.Timestamp.now()
                        })

                # Save to CSV
                csv_file = "classified_data.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)
                else:
                    df = pd.DataFrame(new_records)

                df.to_csv(csv_file, index=False)
                st.success("✅ Saved successfully!")
                st.session_state.expert_input = ""
        else:
            st.warning("⚠️ Please enter a case note before analysis.")
