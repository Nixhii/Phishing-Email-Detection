import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

def generate_synthetic_dataset(n_samples=2000, save=True):
    np.random.seed(42)
    legitimate_emails = [
        "Hey team, please find the quarterly report attached. Let me know if you have any questions.",
        "Your invoice for June has been generated. Total amount due: $150.00.",
        "Meeting reminder: Project sync at 3 PM tomorrow in Conference Room B.",
        "Your package has been shipped and will arrive by Friday. Tracking: 1Z999AA10123456784.",
        "Welcome to our platform! Please verify your email by clicking the link below.",
        "Dear customer, your monthly statement is now available for viewing.",
        "Thank you for your purchase! Your order #12345 has been confirmed.",
        "Reminder: Your subscription will renew on July 1st. Manage your preferences here.",
        "The Q2 financial results are in. Revenue grew by 12% compared to last year.",
        "Please review the updated privacy policy at your earliest convenience.",
        "Your appointment with Dr. Smith is confirmed for Monday at 10:00 AM.",
        "Here is the link to join the webinar on data science best practices.",
        "Your account statement for May 2025 is now ready to view.",
        "Team outing planned for Saturday. Please RSVP by Wednesday.",
        "Your new security pass is ready for collection at the front desk.",
        "Feedback requested: How was your experience with our support team?",
        "The server maintenance is scheduled for Sunday 2 AM - 4 AM. Expected downtime: 2 hours.",
        "Your annual leave balance as of June 2025 is 15 days.",
        "Please complete the mandatory compliance training by end of month.",
        "Congratulations! You have been selected for the next round of interviews.",
        "The board meeting agenda has been updated. Please find the attached document.",
        "Your software license will expire in 30 days. Please renew to continue using the service.",
        "Important update: Our office address has changed effective August 1st.",
        "Your password reset request has been processed successfully.",
        "Thank you for submitting your expense report. It is currently under review.",
        "The weekly stand-up notes are attached. Please review before tomorrow's meeting.",
        "Your application for the research grant has been received and is being evaluated.",
        "Friendly reminder: Please update your emergency contact information.",
        "Your direct deposit has been processed. Payroll will be available by Friday.",
        "The new employee handbook is now available on the intranet.",
    ]
    phishing_emails = [
        "URGENT: Your account has been compromised! Click here immediately to secure your account: http://bit.ly/3xK9mN2",
        "Congratulations! You have won $1,000,000 in the international lottery. Claim your prize now at http://tinyurl.com/claim-prize",
        "Dear valued customer, your Netflix account has been suspended due to payment issues. Update payment: http://netflx-verify.tk",
        "Your email password is about to expire. Keep same password: http://secure-mailbox.xyz/verify",
        "ATTN: IRS Tax Refund of $2,845 is waiting for you. Submit claim: http://irs-refund.tk/claim",
        "Your Apple ID has been locked for security reasons. Verify now: http://apple-id-verify.xyz",
        "You have a new voicemail from (202) 555-0199. Listen here: http://voicemail-msg.tk",
        "Urgent security alert: Someone tried to access your account from Russia. Secure now: http://account-security.tk",
        "Your Amazon order could not be delivered. Update delivery preferences: http://amzn-delivery-update.xyz",
        "Special promotion: Get 90% off on all products today only! Shop now: http://cheap-deals-now.tk",
        "Your Facebook account has been reported for violation. Appeal here: http://facebook-appeal.tk",
        "Your package is waiting at customs. Pay release fee: http://customs-release-payment.xyz",
        "Alert: Unusual login detected on your Google account. Recover now: http://google-recovery.tk",
        "Your PayPal account has been limited. Resolve now: http://paypal-resolve.xyz/verify",
        "COVID-19 relief fund: You are eligible for $2,000 payment. Apply: http://relief-fund-apply.tk",
        "Your Microsoft account has been compromised. Click to reset password: http://microsoft-reset.tk/secure",
        "You have been selected for a free iPhone 15! Claim now: http://free-iphone-winner.tk",
        "Your Bank of America account has been deactivated. Reactivate: http://boa-verify.tk",
        "Dear user, your Dropbox storage is almost full. Upgrade now: http://dropbox-upgrade.tk/offer",
        "Urgent: Your domain is expiring soon. Renew now to avoid loss: http://domain-renewal.tk/pay",
        "Your Instagram account will be deleted due to inactivity. Keep it: http://instagram-save.xyz",
        "You have a refund pending from your last purchase. Process: http://refund-process.tk/bank",
        "Your LinkedIn account has been flagged. Verify your identity: http://linkedin-verify.tk/id",
        "Exclusive investment opportunity: 500% returns guaranteed! http://invest-now-profit.tk",
        "Your utility bill payment failed. Pay now to avoid disconnection: http://bill-pay-urgent.xyz",
        "Your Twitter account has been suspended. Appeal suspension: http://twitter-appeal.tk",
        "Adobe Creative Cloud subscription expiring. Cheap renewal: http://adobe-renew.tk/offer",
        "You have been charged $499.99 for Norton subscription. Dispute: http://norton-charge-dispute.xyz",
        "Your Airbnb account has been compromised. 5 guests booked. Verify: http://airbnb-verify.tk",
        "Your Snapchat account has been hacked. Recover here: http://snapchat-recover.xyz",
    ]
    n_per_class = n_samples // 2
    emails = legitimate_emails * (n_per_class // len(legitimate_emails)) + legitimate_emails[:n_per_class % len(legitimate_emails)]
    labels_legit = [0] * len(emails)
    emails_phishing = phishing_emails * (n_per_class // len(phishing_emails)) + phishing_emails[:n_per_class % len(phishing_emails)]
    labels_phish = [1] * len(emails_phishing)
    emails_final = emails + emails_phishing
    labels_final = labels_legit + labels_phish
    df = pd.DataFrame({"text": emails_final, "label": labels_final})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if save:
        filepath = os.path.join(DATA_PATH, "phishing_dataset.csv")
        df.to_csv(filepath, index=False)
        print(f"[INFO] Synthetic dataset saved to {filepath}")
    return df

def load_dataset(filepath=None):
    if filepath is None:
        filepath = os.path.join(DATA_PATH, "phishing_dataset.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {len(df)} samples from {filepath}")
        return df
    print(f"[INFO] File not found at {filepath}. Generating synthetic dataset...")
    return generate_synthetic_dataset()
