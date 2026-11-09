"""
One-time (or re-runnable) script to load the EZAppointo chatbot knowledge base
from ezappointo_chatbot_1.docx into Qdrant, using the same embedder/client the
running chatbot uses.

Run this from inside the same environment as Chatbot_FS.py (same env vars,
same CHATBOT_DATA_DIR / SENTENCE_TRANSFORMER_MODEL_PATH), so it writes to the
exact same Qdrant collection the API reads from:

    python seed_knowledge_base.py

Safe to re-run — each entry gets a fresh UUID, so re-running just adds
duplicates rather than erroring. If you want a clean reload, delete the
qdrant_storage folder (or the collection) first.
"""

from Chatbot_FS import get_embedding, q_client, COLLECTION_NAME
from qdrant_client.http import models
import uuid

# ── Knowledge chunks pulled from ezappointo_chatbot_1.docx ──────────────────
# One entry per Q&A / feature / topic — keeps each vector focused so search
# (limit=2 in search_knowledge) returns tightly relevant context instead of
# one giant blob.

KNOWLEDGE_ENTRIES = [
    # ── 1. General Knowledge ──────────────────────────────────────────────
    {
        "category": "general",
        "info": "What is EZAppointo? EZAppointo is an all-in-one healthcare appointment "
                "scheduling and practice management platform that helps clinics and "
                "healthcare providers manage appointments, patient records, video "
                "consultations, digital forms, payments, analytics, and more — all from "
                "one secure platform.",
    },
    {
        "category": "general",
        "info": "How does EZAppointo work? EZAppointo allows patients to book appointments "
                "online while providers manage schedules through a centralized dashboard. "
                "It supports secure video consultations, digital patient forms, automated "
                "reminders, online payments, and analytics to streamline daily operations.",
    },
    {
        "category": "general",
        "info": "Who is EZAppointo for? EZAppointo is designed for private practices, "
                "medical clinics, hospitals, healthcare providers, wellness centers, and "
                "multi-provider healthcare organizations. It simplifies appointment "
                "management for both healthcare professionals and patients.",
    },
    {
        "category": "general",
        "info": "EZAppointo features: Online Appointment Scheduling, Built-in Video "
                "Consultations, Digital Patient Forms, Secure Online Payments (Stripe & "
                "PayPal), AI Forecasting & Schedule Projection, EMR Integration, Analytics "
                "& Reporting, AI Booking Voice Agent (Coming Soon), Real-Time "
                "Notifications, Patient Records Management, AI-Powered Language "
                "Translation, Secure File Sharing, Instant Messaging, End-to-End "
                "Encryption.",
    },
    {
        "category": "general",
        "info": "Benefits of EZAppointo: reduces administrative workload, improves patient "
                "experience, increases appointment efficiency, reduces no-shows, supports "
                "paperless workflows, forecasts appointment demand, simplifies telehealth, "
                "enables better business decisions with analytics, improves operational "
                "efficiency, and secures patient information.",
    },
    {
        "category": "general",
        "info": "Pricing: EZAppointo offers flexible pricing plans tailored to different "
                "practice sizes. Contact the sales team for detailed pricing and a "
                "personalized quote. Subscription plans are scalable and suitable for "
                "individual providers, clinics, and larger healthcare organizations, and "
                "can be customized based on business requirements.",
    },
    {
        "category": "general",
        "info": "Security: EZAppointo protects sensitive healthcare information through "
                "end-to-end encryption, secure data handling, encrypted communications, "
                "secure payment processing, protected patient records, and secure file "
                "sharing.",
    },
    {
        "category": "general",
        "info": "Is EZAppointo HIPAA compliant? Yes. EZAppointo is built with HIPAA-ready "
                "workflows to help healthcare providers securely manage patient "
                "information, appointments, telehealth sessions, and medical records "
                "while maintaining privacy and compliance standards.",
    },
    {
        "category": "general",
        "info": "Does EZAppointo work on mobile devices? EZAppointo is web-based and "
                "optimized for desktops, tablets, and smartphones, allowing both providers "
                "and patients to access the platform from virtually any modern device.",
    },
    {
        "category": "general",
        "info": "Does EZAppointo support multiple languages? Yes, EZAppointo includes "
                "AI-powered language translation, enabling patients and providers to "
                "communicate more effectively across different languages during "
                "appointments and virtual consultations.",
    },
    {
        "category": "general",
        "info": "Does EZAppointo support multiple time zones? Yes, EZAppointo supports "
                "scheduling across different time zones, helping practices coordinate "
                "appointments accurately for providers and patients in multiple regions.",
    },

    # ── 2. Patient Knowledge Base ─────────────────────────────────────────
    {
        "category": "patient_faq",
        "info": "How do I create an account? 1. Visit the EZAppointo website. 2. Click "
                "Sign Up. 3. Enter your first name. 4. Enter your last name. 5. Enter "
                "your email address. 6. Create a secure password. 7. Agree to the Terms "
                "of Service and Privacy Policy. 8. Click Create Account. 9. Check your "
                "email inbox for the OTP. 10. Enter your OTP. 11. Return to the login "
                "page. 12. Log in using your email and password. 13. Complete your "
                "profile by adding your phone number, date of birth, and address if "
                "requested. 14. Your account is now ready for booking appointments.",
    },
    {
        "category": "patient_faq",
        "info": "I forgot my password, how do I reset it? 1. Go to the Login page. 2. "
                "Click Forgot Password. 3. Enter your registered email. 4. Click Send "
                "Reset Link/Code. 5. Check your email. 6. Open the password reset email. "
                "7. Click the reset link. 8. Enter your new password. 9. Confirm the new "
                "password. 10. Click Save. 11. Return to the login page. 12. Sign in with "
                "your new password.",
    },
    {
        "category": "patient_faq",
        "info": "How do I book an appointment? 1. Log in. 2. Click Book Appointment. 3. "
                "Search by name, zipcode, doctor, clinic, or professional. 4. View "
                "profile. 5. Check availability. 6. Choose date and time. 7. Select "
                "consultation type. 8. Fill forms. 9. Choose payment method. 10. Confirm "
                "booking. 11. Receive email confirmation. 12. Receive SMS reminder if "
                "enabled. 13. Receive appointment reminder before your scheduled time.",
    },
    {
        "category": "patient_faq",
        "info": "How do I cancel my appointment? 1. Log in to your account. 2. Open "
                "Booking. 3. Select the appointment you wish to cancel. 4. Click Cancel "
                "Appointment. 5. Review any cancellation policy that applies. 6. Confirm "
                "the cancellation. 7. You will receive a confirmation notification by "
                "email, and your appointment status will update.",
    },

    # ── 3. Every Feature Explained ────────────────────────────────────────
    {
        "category": "feature",
        "info": "Appointment Booking lets patients find and book appointments with "
                "trusted, verified healthcare providers, while preventing double-booking "
                "and supporting recurring appointments.",
    },
    {
        "category": "feature",
        "info": "Calendar keeps a provider's schedule organized in real time, showing "
                "available slots, blocked times, and recurring bookings in one view.",
    },
    {
        "category": "feature",
        "info": "Video Consultation enables secure video visits directly within the "
                "platform, so patients and providers don't need any separate tool.",
    },
    {
        "category": "feature",
        "info": "Waiting Room gives providers a dedicated space to admit patients one by "
                "one before a video visit, keeping consultations organized and "
                "professional.",
    },
    {
        "category": "feature",
        "info": "Notifications send real-time alerts about upcoming appointments and "
                "status updates to keep everyone informed.",
    },
    {
        "category": "feature",
        "info": "SMS Reminders send appointment reminders by text message so patients "
                "don't miss their visits.",
    },
    {
        "category": "feature",
        "info": "Email Reminders send email reminders with a one-click link so patients "
                "can join their video consultation instantly.",
    },
    {
        "category": "feature",
        "info": "Digital Forms turn paper intake forms into digital, fillable forms that "
                "patients complete before their appointment, cutting down on paperwork.",
    },
    {
        "category": "feature",
        "info": "Patient Records keep all patient information — intake forms, appointment "
                "history, consultation details — securely stored in one place.",
    },
    {
        "category": "feature",
        "info": "Billing collects consultation fees, deposits, and follow-up payments as "
                "part of the booking process, with all payment activity tracked.",
    },
    {
        "category": "feature",
        "info": "Stripe lets providers accept online payments securely, including "
                "subscriptions and invoicing.",
    },
    {
        "category": "feature",
        "info": "PayPal gives providers an alternative payment option alongside Stripe.",
    },
    {
        "category": "feature",
        "info": "Forecasting predicts busy periods and appointment demand so clinics can "
                "plan staffing and revenue ahead of time.",
    },
    {
        "category": "feature",
        "info": "Dashboard shows providers a clear overview of revenue, bookings, and "
                "clinic performance in one place.",
    },
    {
        "category": "feature",
        "info": "Search lets users search for locations, professionals, doctors, zipcode, "
                "insurance, or services all from a single search bar, making it easy to "
                "find the right care option.",
    },
    {
        "category": "feature",
        "info": "Filters let patients and providers narrow down records by first name, "
                "last name, date range, and status, with control over how many results "
                "show per page — making it easier to quickly find specific entries.",
    },
    {
        "category": "feature",
        "info": "EMR Integration keeps patient records and appointments automatically in "
                "sync with a clinic's existing EMR/EHR system, reducing manual data "
                "entry.",
    },
    {
        "category": "feature",
        "info": "User Profile stores personal details, preferences, and history for each "
                "patient and provider.",
    },

    # ── 5 & 6. Roles ───────────────────────────────────────────────────────
    {
        "category": "roles",
        "info": "How does a Service Manager work? The Service Manager supports the "
                "Service Provider by managing appointments, calendars, and daily "
                "schedules. They coordinate meetings, handle booking requests, organize "
                "availability, and ensure smooth day-to-day operations so the Service "
                "Provider can focus on delivering services.",
    },
    {
        "category": "roles",
        "info": "How does a Service Provider work? The Service Provider delivers services "
                "to customers and manages their assigned appointments. They provide "
                "consultations, complete scheduled services, update appointment details, "
                "and maintain accurate service records to ensure a high-quality customer "
                "experience.",
    },
]


def main() -> None:
    print(f"Seeding {len(KNOWLEDGE_ENTRIES)} entries into Qdrant collection "
          f"'{COLLECTION_NAME}'...")

    points = []
    for entry in KNOWLEDGE_ENTRIES:
        vector = get_embedding(entry["info"])
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"info": entry["info"], "category": entry["category"]},
            )
        )

    q_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Done. {len(points)} entries added to the knowledge base.")


if __name__ == "__main__":
    main()
