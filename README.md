## 1. Problem Statement
The goal of this project is to detect potential insider threats by analyzing user behavior across multiple activity logs. Using the CERT Insider Threat Dataset, we aim to build a system that learns baseline behavior patterns for individual users and flags deviations—such as unusual login times, abnormal file access, or suspicious web activity—as potential security threats.

## 2. Dataset Description
Source: CERT Insider Threat Dataset (Kaggle)[[https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research?select=http.csv]]

## Files Used:
email.csv – Metadata about internal emails (sender, recipients, timestamp, etc.)
http.csv – Web access logs (URL, user, timestamp)
logon.csv – Logon/logoff activity with timestamps and machine IDs
file.csv – File access logs including activity type
users.csv – User profiles (role, department, etc.)
devices – (Custom feature) Activity involving devices, with timestamps

## 3. Tools & Technologies
Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
Platform: Google Colab / Jupyter Notebook
Visualization: Seaborn, Matplotlib

## 4. Feature Engineering Plan
**email.csv**
email_hour (hour of send)
num_recipients (to + cc + bcc)
num_external_recipients
is_off_hours_email (email sent outside 9 AM–5 PM)

**http.csv**
domain_name, tld_type
is_https (flag if not HTTPS)
num_unique_domains_per_day

**logon.csv**
login_hour, day_of_week
is_off_hours_login
is_weekend_login
unique_pcs_used

**file.csv**
file_extension, file_access_hour
is_sensitive_file
file_access_count_per_day

**users.csv**
Used to enrich data for role-activity alignment
Flag users whose behavior doesn’t match their expected role

**devices.csv**
Activity type + timestamp
is_off_hours_device_use flag for night/weekend use

## 5. Custom Anomaly Rules##
File	Anomaly Rule:
1)email.csv	Flag emails with large attachments or suspicious file types
2)http.csv	Flag any web access using non-HTTPS protocol
3)logon.csv	Flag logons occurring outside 9 AM–5 PM or on weekends
3)file.csv	Flag based on suspicious file access (e.g., time, machine, sensitive file)
4)users.csv	Flag based on unexpected role-behavior mismatches
5)devices	Flag device usage with unusual activity types or timestamps
