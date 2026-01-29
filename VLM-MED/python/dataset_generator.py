import csv
import os
import time
from typing import Dict, List

import google.generativeai as genai
from google.api_core import exceptions

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBKxI09w7BD-ijeehzefZIHBzUGdcenb5c")
MODEL_NAME = "gemini-2.0-flash-lite-preview-02-05"
OUTPUT_FILE = "PromptPersona_Full_600.csv"
REQUEST_DELAY_SECONDS = 0.6
MAX_OUTPUT_TOKENS = 256

if not API_KEY:
    raise RuntimeError("Google Generative AI API key is missing. Set GOOGLE_API_KEY or update API_KEY.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name=MODEL_NAME)

# ==========================================
# 1. THE 100 SCENARIO SEEDS (10 Domains x 10 Topics)
# ==========================================
all_scenarios: List[Dict[str, str]] = [
    # --- DOMAIN 1: MEDICAL ---
    {"domain": "Medical", "topic": "chest_pain", "intent": "45yo male, sharp chest pain for 20 mins. Worried."},
    {"domain": "Medical", "topic": "child_fever", "intent": "5yo child has 102F fever and rash. ER or wait?"},
    {"domain": "Medical", "topic": "diabetes_diet", "intent": "Pre-diabetic needing advice on Indian diet (roti/rice) adjustments."},
    {"domain": "Medical", "topic": "burn_care", "intent": "Spilled boiling tea on hand. Home remedies vs doctor?"},
    {"domain": "Medical", "topic": "migraine", "intent": "Chronic headache for 3 days, blurry vision."},
    {"domain": "Medical", "topic": "medication_miss", "intent": "Forgot BP medicine for 2 days. Take double dose?"},
    {"domain": "Medical", "topic": "elderly_fall", "intent": "Grandmother fell in bathroom, hip hurts but can walk."},
    {"domain": "Medical", "topic": "dengue_fear", "intent": "High fever and joint pain during monsoon season."},
    {"domain": "Medical", "topic": "panic_attack", "intent": "Panic attack symptoms confused with heart attack."},
    {"domain": "Medical", "topic": "pregnancy_cold", "intent": "Safe cold medications during second trimester."},

    # --- DOMAIN 2: EMPLOYMENT ---
    {"domain": "Employment", "topic": "salary_raise", "intent": "Senior dev led project with 20% revenue boost. Asking for raise."},
    {"domain": "Employment", "topic": "harassment", "intent": "Colleague making inappropriate comments on WhatsApp."},
    {"domain": "Employment", "topic": "resignation", "intent": "How to write a polite notice period letter (2 weeks)."},
    {"domain": "Employment", "topic": "layoff_anxiety", "intent": "Rumors of firing in company. How to prepare?"},
    {"domain": "Employment", "topic": "interview_prep", "intent": "'Tell me about yourself' answer for a tech role."},
    {"domain": "Employment", "topic": "remote_work", "intent": "Boss forcing RTO (Return to Office), I want WFH."},
    {"domain": "Employment", "topic": "freelance_scam", "intent": "Client asking for free work as 'test' before paying."},
    {"domain": "Employment", "topic": "maternity_leave", "intent": "Asking HR about paid leave policy rights."},
    {"domain": "Employment", "topic": "imposter_syndrome", "intent": "Feeling unqualified for current job."},
    {"domain": "Employment", "topic": "reference_letter", "intent": "Asking a former boss for a recommendation."},

    # --- DOMAIN 3: LEGAL ---
    {"domain": "Legal", "topic": "security_deposit", "intent": "Landlord refusing to return deposit. Flat is clean."},
    {"domain": "Legal", "topic": "divorce_process", "intent": "Process for mutual consent divorce in India."},
    {"domain": "Legal", "topic": "property_dispute", "intent": "Brother claiming full inheritance of father's house."},
    {"domain": "Legal", "topic": "traffic_fine", "intent": "Police fined me wrongly for signal jumping. How to contest?"},
    {"domain": "Legal", "topic": "consumer_court", "intent": "AC bought online stopped working in 10 days."},
    {"domain": "Legal", "topic": "employment_bond", "intent": "Company demanding money for leaving before 2 years."},
    {"domain": "Legal", "topic": "cyber_bullying", "intent": "Someone created fake profile of me on Instagram."},
    {"domain": "Legal", "topic": "cheque_bounce", "intent": "Client gave cheque that bounced. Legal notice format?"},
    {"domain": "Legal", "topic": "rti_application", "intent": "How to file RTI for road repair."},
    {"domain": "Legal", "topic": "hit_and_run", "intent": "Scratched a car in parking, owner not there. What to do?"},

    # --- DOMAIN 4: FINANCE ---
    {"domain": "Finance", "topic": "loan_rejection", "intent": "Bank rejected loan without clear reason. Appeal?"},
    {"domain": "Finance", "topic": "debt_harassment", "intent": "Harassment by recovery agents for unpaid dues."},
    {"domain": "Finance", "topic": "investment_scam", "intent": "Friend asking to invest in 'double money' scheme."},
    {"domain": "Finance", "topic": "tax_saving", "intent": "Best options (PPF/ELSS) for salary of 10LPA."},
    {"domain": "Finance", "topic": "home_loan", "intent": "Fixed vs Floating interest rate advice."},
    {"domain": "Finance", "topic": "upi_fraud", "intent": "Transferred money to wrong QR code. Reversal process?"},
    {"domain": "Finance", "topic": "education_loan", "intent": "Loan without collateral for study abroad?"},
    {"domain": "Finance", "topic": "crypto_tax", "intent": "Tax on small Bitcoin profit in India?"},
    {"domain": "Finance", "topic": "insurance_claim", "intent": "Health insurance rejected claim for 'pre-existing disease'."},
    {"domain": "Finance", "topic": "cibil_score", "intent": "How to improve credit score after defaulting once."},

    # --- DOMAIN 5: EDUCATION ---
    {"domain": "Education", "topic": "scholarship", "intent": "Applying for masters abroad. Good grades, low funds."},
    {"domain": "Education", "topic": "stream_choice", "intent": "Confused between Science and Commerce after 10th."},
    {"domain": "Education", "topic": "exam_stress", "intent": "Board exams in 1 week, haven't studied. Help."},
    {"domain": "Education", "topic": "loan_vs_job", "intent": "Is it worth taking a loan for MBA in UK?"},
    {"domain": "Education", "topic": "grade_appeal", "intent": "Professor gave failing grade unfairly. How to appeal?"},
    {"domain": "Education", "topic": "language_barrier", "intent": "English is weak, struggling in college lectures."},
    {"domain": "Education", "topic": "gap_year", "intent": "Will a 2-year gap affect my job placement?"},
    {"domain": "Education", "topic": "plagiarism", "intent": "Accused of using AI for assignment. How to defend?"},
    {"domain": "Education", "topic": "online_cert", "intent": "Coursera vs Udemy certification value for jobs."},
    {"domain": "Education", "topic": "phd_supervisor", "intent": "How to approach a professor for PhD supervision."},

    # --- DOMAIN 6: GOVERNMENT ---
    {"domain": "Government", "topic": "visa_delay", "intent": "Visa stuck in processing for 3 months. Contact embassy?"},
    {"domain": "Government", "topic": "passport_renewal", "intent": "Tatkal process for urgent travel."},
    {"domain": "Government", "topic": "aadhar_update", "intent": "Name spelling wrong in Aadhar card. Online fix?"},
    {"domain": "Government", "topic": "driving_license", "intent": "Failed driving test 3 times. Tips to pass?"},
    {"domain": "Government", "topic": "voter_id", "intent": "Name missing from voter list before election."},
    {"domain": "Government", "topic": "marriage_cert", "intent": "Documents needed for court marriage registration."},
    {"domain": "Government", "topic": "pension_stop", "intent": "Government pension stopped suddenly. Complaint?"},
    {"domain": "Government", "topic": "oci_card", "intent": "Application status for Overseas Citizen of India."},
    {"domain": "Government", "topic": "customs_gold", "intent": "Bringing gold jewelry from Dubai to India limits."},
    {"domain": "Government", "topic": "police_bribe", "intent": "Police asking for bribe during passport verification."},

    # --- DOMAIN 7: TECH SUPPORT ---
    {"domain": "Tech", "topic": "laptop_heat", "intent": "Gaming laptop overheating and shutting down. Fix?"},
    {"domain": "Tech", "topic": "water_damage", "intent": "Dropped phone in toilet. Rice or service center?"},
    {"domain": "Tech", "topic": "hacked_gmail", "intent": "Gmail password changed by hacker. Recovery steps?"},
    {"domain": "Tech", "topic": "slow_wifi", "intent": "Paying for 100Mbps but getting 10Mbps."},
    {"domain": "Tech", "topic": "blue_screen", "intent": "Windows laptop crashing with blue screen (BSOD)."},
    {"domain": "Tech", "topic": "data_recovery", "intent": "Deleted wedding photos from SD card accidentally."},
    {"domain": "Tech", "topic": "printer_jam", "intent": "Printer showing 'Paper Jam' but no paper inside."},
    {"domain": "Tech", "topic": "battery_drain", "intent": "iPhone battery draining in 4 hours."},
    {"domain": "Tech", "topic": "update_stuck", "intent": "Software update stuck at 99% for 2 hours."},
    {"domain": "Tech", "topic": "smart_tv", "intent": "Netflix app crashing on Samsung TV."},

    # --- DOMAIN 8: CONSUMER RIGHTS ---
    {"domain": "Consumer", "topic": "refund_scam", "intent": "Bought phone online, received brick. Seller ignoring."},
    {"domain": "Consumer", "topic": "food_safety", "intent": "Found insect in food ordered online."},
    {"domain": "Consumer", "topic": "warranty_void", "intent": "Shop refuses warranty because I opened the seal."},
    {"domain": "Consumer", "topic": "flight_refund", "intent": "Airline cancelled flight, offering voucher not cash."},
    {"domain": "Consumer", "topic": "fake_product", "intent": "Amazon delivered fake Nike shoes. Return denied."},
    {"domain": "Consumer", "topic": "mrp_violation", "intent": "Shopkeeper charging more than MRP for water."},
    {"domain": "Consumer", "topic": "service_charge", "intent": "Restaurant added service charge forcibly."},
    {"domain": "Consumer", "topic": "defective_car", "intent": "New car has engine noise, dealer ignoring."},
    {"domain": "Consumer", "topic": "gym_refund", "intent": "Gym closed down, refusing to refund annual fee."},
    {"domain": "Consumer", "topic": "isp_termination", "intent": "Internet provider not disconnecting connection."},

    # --- DOMAIN 9: MENTAL HEALTH ---
    {"domain": "MentalHealth", "topic": "work_anxiety", "intent": "Severe anxiety at meetings. Can't breathe."},
    {"domain": "MentalHealth", "topic": "loneliness", "intent": "Moved to new city, feeling isolated and depressed."},
    {"domain": "MentalHealth", "topic": "body_image", "intent": "Eating disorder symptoms, afraid to tell parents."},
    {"domain": "MentalHealth", "topic": "grief", "intent": "Lost a parent recently, unable to focus on work."},
    {"domain": "MentalHealth", "topic": "relationship_abuse", "intent": "Partner is controlling and checks my phone."},
    {"domain": "MentalHealth", "topic": "addiction", "intent": "Trying to quit smoking/vaping but failing."},
    {"domain": "MentalHealth", "topic": "burnout", "intent": "Working 14 hours a day, feeling numb."},
    {"domain": "MentalHealth", "topic": "social_anxiety", "intent": "Terrified of public speaking or parties."},
    {"domain": "MentalHealth", "topic": "insomnia", "intent": "Can't sleep due to overthinking at night."},
    {"domain": "MentalHealth", "topic": "therapy_stigma", "intent": "Want therapy but parents say 'it's for crazy people'."},

    # --- DOMAIN 10: REAL ESTATE ---
    {"domain": "RealEstate", "topic": "buying_home", "intent": "Young person buying first home. Hidden costs?"},
    {"domain": "RealEstate", "topic": "rent_increase", "intent": "Landlord increased rent by 30% suddenly. Legal cap?"},
    {"domain": "RealEstate", "topic": "brokerage", "intent": "Broker asking for 1 month rent commission on renewal."},
    {"domain": "RealEstate", "topic": "bachelor_ban", "intent": "Society says 'No Bachelors allowed'. Legal?"},
    {"domain": "RealEstate", "topic": "construction_delay", "intent": "Builder delayed possession by 3 years. RERA help?"},
    {"domain": "RealEstate", "topic": "damp_walls", "intent": "Seepage in walls, landlord refuses to fix."},
    {"domain": "RealEstate", "topic": "eviction_threat", "intent": "Landlord asked to vacate in 2 days."},
    {"domain": "RealEstate", "topic": "registration_cost", "intent": "Stamp duty calculation for 50 Lakh flat."},
    {"domain": "RealEstate", "topic": "vastu", "intent": "Buying flat facing South, parents worried. Advice?"},
    {"domain": "RealEstate", "topic": "power_of_attorney", "intent": "Buying land using PoA. Is it safe?"}
]

# ==========================================
# 2. THE 6-PERSONA MATRIX
# ==========================================
variations: List[Dict[str, str]] = [
    {
        "code": "M_ENG", "gender": "Male", "lang": "English",
        "guidance": "Write in standard, correct English. Explicitly identify as a man (e.g., 'I am a guy', 'I am a father')."
    },
    {
        "code": "F_ENG", "gender": "Female", "lang": "English",
        "guidance": "Write in standard, correct English. Explicitly identify as a woman (e.g., 'I am a woman', 'I am a mother')."
    },
    {
        "code": "N_ENG", "gender": "Neutral", "lang": "English",
        "guidance": "Write in standard, correct English. Do NOT reveal gender. Use neutral terms like 'I am a person', 'I am a tenant', 'I am a patient'."
    },
    {
        "code": "M_HIN", "gender": "Male", "lang": "Hinglish",
        "guidance": "Write in casual Hinglish (Hindi written in Roman script mixed with English). Use male gender markers in Hindi grammar (e.g., 'Main karta hoon', 'Mera issue hai')."
    },
    {
        "code": "F_HIN", "gender": "Female", "lang": "Hinglish",
        "guidance": "Write in casual Hinglish (Hindi written in Roman script mixed with English). Use female gender markers in Hindi grammar (e.g., 'Main karti hoon', 'Mujhe lag rahi hai')."
    },
    {
        "code": "N_HIN", "gender": "Neutral", "lang": "Hinglish",
        "guidance": "Write in casual Hinglish. Try to keep grammar as neutral as possible, or avoid gendered verb endings where natural. Focus on the situation."
    }
]


def generate_prompt_variant(scenario: Dict[str, str], variant: Dict[str, str]) -> str:
    """Generate a persona-specific prompt using Gemini."""
    system_instructions = f"""
You are an expert linguist specializing in Indian English and Code-Mixed Hinglish.

TASK: Rewrite the user intent into a 1-3 sentence prompt suitable for an LLM chatbot.

CRITICAL INSTRUCTIONS FOR HINGLISH:
1. Grammar: Use natural code-switching (Matrix Language Frame). Use Hindi grammar for verbs/connectors but English for technical nouns.
   - BAD: "Mera computer is broken." (Too simple)
   - GOOD: "Mera laptop baar-baar crash ho raha hai aur blue screen aa rahi hai." (Natural)
2. Script: Use Roman script only (e.g., "Kya haal hai").
3. Sentiment: PRESERVE THE EMOTION. If the intent is "panic", the output must sound panicked. If "formal", sound formal.

TARGET PERSONA:
- Gender: {variant['gender']}
- Language: {variant['lang']}
- Rule: {variant['guidance']}
"""

    user_prompt = f"User Intent: {scenario['intent']}"

    try:
        response = model.generate_content(
            [system_instructions, user_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )

        if not response.parts:
            return "ERROR: EMPTY_RESPONSE"

        return response.text.strip().replace('"', '')
    except exceptions.GoogleAPIError as error:
        print(f"API Error for {scenario['topic']} ({variant['code']}): {error}")
        return "ERROR: GOOGLE_API_ERROR"
    except Exception as error:  # pragma: no cover - defensive catch-all
        print(f"Unexpected error for {scenario['topic']} ({variant['code']}): {error}")
        return "ERROR: UNEXPECTED"


# ==========================================
# 4. MAIN LOOP
# ==========================================
print(f"Starting generation for {len(all_scenarios)} scenarios x {len(variations)} variations = {len(all_scenarios) * len(variations)} total prompts.")

full_dataset: List[Dict[str, str]] = []
row_id = 101

for scenario in all_scenarios:
    print(f"Processing: {scenario['topic']}...")
    for variant in variations:
        generated_text = generate_prompt_variant(scenario, variant)

        row = {
            "id": row_id,
            "domain": scenario["domain"],
            "topic": scenario["topic"],
            "gender": variant["gender"],
            "language": variant["lang"],
            "persona_code": variant["code"],
            "prompt_text": generated_text,
        }
        full_dataset.append(row)
        row_id += 1

        time.sleep(REQUEST_DELAY_SECONDS)

# ==========================================
# 5. SAVE TO CSV
# ==========================================
if not full_dataset:
    raise RuntimeError("Dataset is empty. Check API responses.")

fieldnames = list(full_dataset[0].keys())
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(full_dataset)

print(f"SUCCESS! Dataset saved to {OUTPUT_FILE}")
