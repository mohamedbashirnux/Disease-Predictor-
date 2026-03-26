from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = DATA_DIR / "disease_symptoms.csv"


# You can safely add more symptoms here (snake_case).
SYMPTOMS = [
    "fever",
    "cough",
    "fatigue",
    "difficulty_breathing",
    "headache",
    "sore_throat",
    "runny_nose",
    "body_ache",
    "chills",
    "nausea",
    "vomiting",
    "diarrhea",
    "chest_pain",
    "loss_of_taste",
    "loss_of_smell",
    "muscle_pain",
    "joint_pain",
    "sweating",
    "rash",
    "weight_loss",
    "night_sweats",
    "wheezing",
    "sneezing",
    "increased_thirst",
    "frequent_urination",
    "blurred_vision",
    "abdominal_pain",
    "loss_of_appetite",
    "dizziness",
    "yellowing_eyes",
    "dark_urine",
    "itching",
    "watery_eyes",
    "stuffy_nose",
    "ear_pain",
    "back_pain",
    "burning_urination",
    "shortness_of_breath",
    "heartburn",
    # New symptoms for additional diseases
    "swollen_lymph_nodes",
    "mouth_sores",
    "persistent_cough",
    "blood_in_stool",
    "blood_in_urine",
    "difficulty_swallowing",
    "hoarseness",
    "lumps",
    "unexplained_bruising",
    "pale_skin",
    "rapid_heartbeat",
    "cold_hands_feet",
    "brittle_nails",
    "swollen_joints",
    "stiffness",
    "red_eyes",
    "sensitivity_to_light",
    "confusion",
    "memory_loss",
    "tremors",
    "seizures",
    "numbness",
    "tingling",
    "weakness",
    "paralysis",
    "slurred_speech",
    "vision_loss",
    "double_vision",
    "anxiety",
    "depression",
    "insomnia",
    "excessive_sleepiness",
    "mood_swings",
    "irritability",
    "constipation",
    "bloating",
    "gas",
    "blood_vomiting",
    "black_stool",
    "yellow_skin",
    "swollen_abdomen",
    "leg_swelling",
    "dry_skin",
    "hair_loss",
    "cold_intolerance",
    "heat_intolerance",
    "excessive_hunger",
    "slow_healing",
    "recurrent_infections",
    "dry_mouth",
    "bad_breath",
    "breast_pain",
    "nipple_discharge",
    "difficulty_urination",
    "bone_pain",
    "nosebleed",
    "difficulty_walking",
    "difficulty_speaking",
    "balance_problems",
    "loss_of_consciousness",
    "staring_spells",
    "muscle_jerking",
    "stiff_neck",
    "muscle_cramps",
    "severe_back_pain",
    "weight_gain",
    "chest_tightness",
    "sadness",
    "loss_of_interest",
    "sleep_changes",
    "appetite_changes",
    "difficulty_concentrating",
    "suicidal_thoughts",
    "excessive_worry",
    "restlessness",
    "red_patches",
    "scaling",
    "nail_changes",
    "red_skin",
    "swelling",
    "crusting",
    "swollen_glands",
    "slow_movement",
    "frequent_infections",
]


# Disease profiles: "core" symptoms usually present, "optional" sometimes present.
# Add more diseases by adding a new entry here.
DISEASE_PROFILES: dict[str, dict[str, list[str]]] = {
    "Flu": {
        "core": ["fever", "cough", "fatigue", "body_ache", "chills", "headache"],
        "optional": ["sore_throat", "sweating"],
    },
    "Common Cold": {
        "core": ["cough", "sore_throat", "runny_nose", "sneezing", "stuffy_nose"],
        "optional": ["headache", "fatigue"],
    },
    "COVID-19": {
        "core": ["fever", "cough", "fatigue", "loss_of_taste", "loss_of_smell"],
        "optional": ["difficulty_breathing", "shortness_of_breath", "body_ache"],
    },
    "Pneumonia": {
        "core": ["fever", "cough", "chest_pain", "difficulty_breathing", "fatigue"],
        "optional": ["shortness_of_breath", "chills"],
    },
    "Asthma": {
        "core": ["wheezing", "shortness_of_breath", "cough"],
        "optional": ["chest_pain", "fatigue"],
    },
    "Migraine": {
        "core": ["headache", "nausea"],
        "optional": ["vomiting", "fatigue", "dizziness", "blurred_vision"],
    },
    "Dengue": {
        "core": ["fever", "headache", "rash", "joint_pain", "muscle_pain"],
        "optional": ["nausea", "vomiting"],
    },
    "Malaria": {
        "core": ["fever", "chills", "sweating", "fatigue", "headache"],
        "optional": ["nausea", "vomiting"],
    },
    "Typhoid": {
        "core": ["fever", "fatigue", "abdominal_pain", "loss_of_appetite"],
        "optional": ["diarrhea", "headache"],
    },
    "Tuberculosis": {
        "core": ["persistent_cough", "weight_loss", "night_sweats", "fatigue"],
        "optional": ["fever", "chest_pain", "blood_in_stool"],
    },
    "Diabetes Type 2": {
        "core": ["increased_thirst", "frequent_urination", "fatigue", "excessive_hunger"],
        "optional": ["blurred_vision", "weight_loss", "slow_healing"],
    },
    "Allergy": {
        "core": ["sneezing", "runny_nose", "watery_eyes", "itching", "stuffy_nose"],
        "optional": ["cough"],
    },
    "UTI": {
        "core": ["burning_urination", "frequent_urination", "abdominal_pain"],
        "optional": ["fever", "back_pain"],
    },
    "Gastritis": {
        "core": ["abdominal_pain", "nausea", "heartburn", "loss_of_appetite"],
        "optional": ["vomiting"],
    },
    "Hepatitis": {
        "core": ["fatigue", "loss_of_appetite", "yellowing_eyes", "dark_urine", "yellow_skin"],
        "optional": ["abdominal_pain", "nausea", "itching"],
    },
    "Ear Infection": {
        "core": ["ear_pain", "fever"],
        "optional": ["headache", "sore_throat"],
    },
    # HIV/AIDS
    "HIV/AIDS": {
        "core": ["fever", "fatigue", "swollen_lymph_nodes", "weight_loss", "night_sweats"],
        "optional": ["rash", "mouth_sores", "recurrent_infections"],
    },
    # Cancers
    "Lung Cancer": {
        "core": ["persistent_cough", "chest_pain", "shortness_of_breath", "weight_loss"],
        "optional": ["cough", "fatigue", "hoarseness", "blood_in_stool"],
    },
    "Breast Cancer": {
        "core": ["lumps", "breast_pain", "nipple_discharge"],
        "optional": ["weight_loss", "fatigue", "swollen_lymph_nodes"],
    },
    "Colon Cancer": {
        "core": ["blood_in_stool", "abdominal_pain", "weight_loss", "diarrhea"],
        "optional": ["constipation", "fatigue", "bloating"],
    },
    "Prostate Cancer": {
        "core": ["frequent_urination", "difficulty_urination", "blood_in_urine"],
        "optional": ["back_pain", "bone_pain", "weight_loss"],
    },
    "Leukemia": {
        "core": ["fatigue", "fever", "unexplained_bruising", "recurrent_infections"],
        "optional": ["weight_loss", "swollen_lymph_nodes", "night_sweats"],
    },
    # Heart & Circulatory
    "Heart Disease": {
        "core": ["chest_pain", "shortness_of_breath", "fatigue"],
        "optional": ["dizziness", "nausea", "sweating"],
    },
    "Hypertension": {
        "core": ["headache", "dizziness", "blurred_vision"],
        "optional": ["chest_pain", "shortness_of_breath", "nosebleed"],
    },
    "Anemia": {
        "core": ["fatigue", "weakness", "pale_skin", "dizziness"],
        "optional": ["cold_hands_feet", "brittle_nails", "rapid_heartbeat"],
    },
    "Stroke": {
        "core": ["weakness", "numbness", "slurred_speech", "confusion"],
        "optional": ["headache", "vision_loss", "dizziness"],
    },
    # Autoimmune
    "Rheumatoid Arthritis": {
        "core": ["joint_pain", "swollen_joints", "stiffness", "fatigue"],
        "optional": ["fever", "weight_loss"],
    },
    "Lupus": {
        "core": ["fatigue", "joint_pain", "rash", "fever"],
        "optional": ["chest_pain", "headache", "hair_loss"],
    },
    "Multiple Sclerosis": {
        "core": ["numbness", "weakness", "vision_loss", "fatigue"],
        "optional": ["dizziness", "tremors", "difficulty_walking"],
    },
    # Neurological
    "Alzheimer's Disease": {
        "core": ["memory_loss", "confusion", "difficulty_speaking"],
        "optional": ["mood_swings", "depression", "anxiety"],
    },
    "Parkinson's Disease": {
        "core": ["tremors", "stiffness", "slow_movement"],
        "optional": ["balance_problems", "depression", "constipation"],
    },
    "Epilepsy": {
        "core": ["seizures", "loss_of_consciousness"],
        "optional": ["confusion", "staring_spells", "muscle_jerking"],
    },
    "Meningitis": {
        "core": ["fever", "headache", "stiff_neck", "sensitivity_to_light"],
        "optional": ["nausea", "vomiting", "confusion"],
    },
    # Digestive
    "Crohn's Disease": {
        "core": ["abdominal_pain", "diarrhea", "weight_loss", "fatigue"],
        "optional": ["fever", "blood_in_stool", "loss_of_appetite"],
    },
    "Ulcerative Colitis": {
        "core": ["diarrhea", "blood_in_stool", "abdominal_pain"],
        "optional": ["weight_loss", "fatigue", "fever"],
    },
    "IBS": {
        "core": ["abdominal_pain", "bloating", "diarrhea", "constipation"],
        "optional": ["gas", "nausea"],
    },
    "Celiac Disease": {
        "core": ["diarrhea", "abdominal_pain", "bloating", "weight_loss"],
        "optional": ["fatigue", "nausea", "vomiting"],
    },
    "Cirrhosis": {
        "core": ["fatigue", "yellowing_eyes", "yellow_skin", "swollen_abdomen"],
        "optional": ["itching", "leg_swelling", "confusion"],
    },
    "Pancreatitis": {
        "core": ["abdominal_pain", "nausea", "vomiting", "fever"],
        "optional": ["weight_loss", "diarrhea"],
    },
    # Kidney & Urinary
    "Kidney Disease": {
        "core": ["fatigue", "leg_swelling", "frequent_urination", "nausea"],
        "optional": ["loss_of_appetite", "itching", "muscle_cramps"],
    },
    "Kidney Stones": {
        "core": ["severe_back_pain", "blood_in_urine", "nausea"],
        "optional": ["fever", "frequent_urination", "burning_urination"],
    },
    # Thyroid
    "Hypothyroidism": {
        "core": ["fatigue", "weight_gain", "cold_intolerance", "constipation"],
        "optional": ["dry_skin", "hair_loss", "depression"],
    },
    "Hyperthyroidism": {
        "core": ["weight_loss", "rapid_heartbeat", "anxiety", "heat_intolerance"],
        "optional": ["tremors", "sweating", "insomnia"],
    },
    # Respiratory
    "COPD": {
        "core": ["shortness_of_breath", "persistent_cough", "wheezing", "chest_tightness"],
        "optional": ["fatigue", "frequent_infections"],
    },
    "Bronchitis": {
        "core": ["cough", "chest_pain", "fatigue", "shortness_of_breath"],
        "optional": ["fever", "chills", "body_ache"],
    },
    # Mental Health
    "Depression": {
        "core": ["sadness", "loss_of_interest", "fatigue", "sleep_changes"],
        "optional": ["appetite_changes", "difficulty_concentrating", "suicidal_thoughts"],
    },
    "Anxiety Disorder": {
        "core": ["excessive_worry", "restlessness", "rapid_heartbeat"],
        "optional": ["sweating", "tremors", "insomnia"],
    },
    # Skin
    "Psoriasis": {
        "core": ["red_patches", "itching", "dry_skin", "scaling"],
        "optional": ["joint_pain", "nail_changes"],
    },
    "Eczema": {
        "core": ["itching", "red_skin", "dry_skin", "rash"],
        "optional": ["swelling", "crusting"],
    },
    # Infectious
    "Chickenpox": {
        "core": ["rash", "fever", "itching", "fatigue"],
        "optional": ["headache", "loss_of_appetite"],
    },
    "Measles": {
        "core": ["fever", "cough", "rash", "red_eyes"],
        "optional": ["runny_nose", "sore_throat"],
    },
    "Mumps": {
        "core": ["swollen_glands", "fever", "headache", "muscle_pain"],
        "optional": ["loss_of_appetite", "fatigue"],
    },
    "Mononucleosis": {
        "core": ["fever", "sore_throat", "swollen_lymph_nodes", "fatigue"],
        "optional": ["rash", "headache", "loss_of_appetite"],
    },
}


def generate_dataset(rows_per_disease: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    all_symptoms = list(dict.fromkeys(SYMPTOMS))
    symptom_index = {s: i for i, s in enumerate(all_symptoms)}

    records: list[dict[str, int | str]] = []

    for disease, profile in DISEASE_PROFILES.items():
        core = profile.get("core", [])
        optional = profile.get("optional", [])

        # Guard against typos in symptom names
        unknown = [s for s in (core + optional) if s not in symptom_index]
        if unknown:
            raise ValueError(f"Disease '{disease}' has unknown symptom(s): {unknown}")

        for _ in range(rows_per_disease):
            x = np.zeros(len(all_symptoms), dtype=np.int8)

            # Core symptoms: very likely (increased from 0.85 to 0.92)
            for s in core:
                if rng.random() < 0.92:
                    x[symptom_index[s]] = 1

            # Optional symptoms: sometimes (reduced from 0.35 to 0.25)
            for s in optional:
                if rng.random() < 0.25:
                    x[symptom_index[s]] = 1

            # Noise: random unrelated symptoms (reduced from 0.10 to 0.05)
            noise_count = int(rng.integers(0, 2))
            if noise_count > 0:
                noise_symptoms = rng.choice(all_symptoms, size=noise_count, replace=False)
                for s in noise_symptoms:
                    if rng.random() < 0.05:
                        x[symptom_index[s]] = 1

            row: dict[str, int | str] = {sym: int(x[i]) for sym, i in symptom_index.items()}
            row["Disease"] = disease
            records.append(row)

    df = pd.DataFrame.from_records(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle rows
    return df


if __name__ == "__main__":
    df = generate_dataset(rows_per_disease=120, seed=42)  # Increased from 80 to 120
    df.to_csv(OUT_PATH, index=False)

    print("Dataset created successfully!")
    print(f"\nSaved to: {OUT_PATH}")
    print(f"Dataset shape: {df.shape[0]} samples, {df.shape[1]-1} symptoms")
    print(f"Total Diseases: {df['Disease'].nunique()}")
    print("\nDiseases included:")
    for i, disease in enumerate(sorted(df["Disease"].unique()), 1):
        count = int((df["Disease"] == disease).sum())
        print(f"  {i}. {disease} ({count} samples)")

    print(f"\nTotal Symptoms: {df.shape[1]-1}")
    print("Symptoms list:")
    for i, symptom in enumerate([c for c in df.columns if c != "Disease"], 1):
        print(f"  {i}. {symptom.replace('_', ' ').title()}")
