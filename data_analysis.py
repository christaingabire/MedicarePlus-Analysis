import pandas as pd
import sqlite3
import ace_tools_open as tools

conn = sqlite3.connect('healthcare.db')

patients = pd.read_sql('SELECT * FROM patients', conn)
providers = pd.read_sql('SELECT * FROM providers', conn)
procedures = pd.read_sql('SELECT * FROM procedures', conn)
claims = pd.read_sql('SELECT * FROM claims', conn)

# DATA CLEANING AND PREPARATION

    # Show first 5 rows
print(patients.head())  
print(providers.head())
print(procedures.head())
print(claims.head())

    # Check for missing values
print("Missing values in patients: \n", patients.isnull().sum())
print("Missing values in provider: \n", providers.isnull().sum())
print("Missing values in procedures:\n", procedures.isnull().sum())
print("Missing values in claims:\n", claims.isnull().sum())

    # Fill missing billed_amount with the average billed amount
claims["billed_amount"].fillna(claims["billed_amount"].mean(), inplace=True)

print("Missing values in claims after fixing:\n", claims.isnull().sum())

    # Check for duplicate rows
print("Duplicate rows in patients:", patients.duplicated().sum())
print("Duplicate rows in providers:", providers.duplicated().sum())
print("Duplicate rows in procedures:", procedures.duplicated().sum())
print("Duplicate rows in claims:", claims.duplicated().sum())

    # Check for negative billed amount
print("Negative billed amount: \n", claims[claims["billed_amount"] < 0])

    # Fix negative billed amounts
claims["billed_amount"] = claims["billed_amount"].abs()

print("Negative billed amounts after fixing:\n", claims[claims["billed_amount"] < 0])

    # Final summary statistics
print(claims["billed_amount"].describe())

    # Extra check: Negative paid amounts?
print("Negative paid amount:\n", claims[claims["paid_amount"] < 0])


#EXPLORATORY DATA ANALYSIS (EDA)
    
    # Total and Average billed amount and paid amount
print("\n TOTAL AND AVERAGE CLAIM COSTS \n")
print("Total billed amount: $", claims["billed_amount"].sum())
print("Total paid amount: $", claims["paid_amount"].sum())
print("Average billed amount: $", claims["billed_amount"].mean())
print("Average paid amount: $", claims["paid_amount"].mean())

    # Claims filed
print("\n TOTAL NUMBER OF CLAIMS FILED \n")
print(claims.shape[0])

    # Claims per patient
print("\n TOP 10 PATIENTS WITH MOST CLAIMS \n")
claims_per_patient = claims.groupby("patient_id").size().reset_index(name="num_claims")
claims_per_patient = claims_per_patient.merge(patients[["patient_id", "name"]], on="patient_id", how="left")
claims_per_patient = claims_per_patient[["name", "patient_id", "num_claims"]]
print(claims_per_patient.sort_values(by="num_claims", ascending=False).head(10).to_string(index=False))

    # Claims per provider 
print("\n TOP 10 PROVIDERS WITH MOST CLAIMS \n")
claims_per_provider = claims.groupby("provider_id").size().reset_index(name="num_claims")
claims_per_provider = claims_per_provider.merge(providers[["provider_id", "name", "specialty"]], on="provider_id", how="left")
claims_per_provider = claims_per_provider[["name", "specialty", "provider_id", "num_claims"]]
print(claims_per_provider.sort_values(by="num_claims", ascending=False).head(10).to_string(index=False))

    # Claims per specialty (as a percentage)
print("\n CLAIMS PERCENTAGE BY SPECIALTY \n")
claims_by_specialty = claims_per_provider["specialty"].value_counts(normalize=True) * 100
print(claims_by_specialty.to_string())

    # Claims per procedure
print("\n TOP 10 MOST COMMON PROCEDURES \n ")
claims_per_procedure = claims.groupby("procedure_id").size().reset_index(name="num_claims")
claims_per_procedure = claims_per_procedure.merge(procedures[["procedure_id", "description"]], on="procedure_id", how="left")
claims_per_procedure = claims_per_procedure[["description", "procedure_id", "num_claims"]]
print(claims_per_procedure.sort_values(by="num_claims", ascending=False).head(10).to_string(index=False))

print("\n TOP 10 MOST EXPENSIVE PROCEDURES \n ")
top_expensive_procedures = claims.merge(procedures, on="procedure_id", how="left")
top_expensive_procedures = top_expensive_procedures.groupby(["description"])["billed_amount"].sum().reset_index()
top_expensive_procedures = top_expensive_procedures.sort_values(by="billed_amount", ascending=False).head(10)
top_expensive_procedures["billed_amount"] = top_expensive_procedures["billed_amount"].apply(lambda x: f"${x:,.2f}")  # Format billed_amount to show currency symbol
print(top_expensive_procedures.to_string(index=False))

    # Cost by age group
print("\n AVERAGE BILLED AMOUNT BY AGE GROUP \n")
bins = [0, 18, 26, 35, 50, 70, 100]
labels = ["<18", "18-26", "26-35", "35-50", "50-70", "70+"]
claims_patients = claims.merge(patients, on="patient_id", how="left")
claims_patients["age_group"] = pd.cut(claims_patients["age"], bins=bins, labels=labels, right=False)

    # Average billed amount per age group
cost_by_age = claims_patients.groupby("age_group", observed=False)["billed_amount"].mean().reset_index()
cost_by_age["billed_amount"] = cost_by_age["billed_amount"].fillna(0)
print(cost_by_age.to_string(index=False))

    # Claim approval rate (billed vs. paid)
approval_rate = (claims["paid_amount"].sum() / claims["billed_amount"].sum()) * 100
print(f"\n Overall Claim Approval Rate: {approval_rate:.2f}%")

# FRAUD DETECTION ANALYSIS

    # Identify providers with high denial rates
claims['claim_status'] = claims['paid_amount'].apply(lambda x: "Denied" if x == 0 else "Approved")

provider_claims = claims.groupby('provider_id').agg(
    total_claims=('claim_id', 'count'),
    denied_claims=('claim_status', lambda x: (x == "Denied").sum())
).reset_index()

provider_claims['denial_rate'] = (provider_claims['denied_claims'] / provider_claims['total_claims']) * 100
provider_claims = provider_claims.merge(providers[['provider_id', 'name']], on='provider_id', how='left')

high_denial_providers = provider_claims[provider_claims['denial_rate'] > 50]

    # Identify high billed patients
patient_claims = claims.groupby('patient_id').agg(
    total_billed=('billed_amount', 'sum'),
    total_claims=('claim_id', 'count')
).reset_index()

patient_claims = patient_claims.merge(patients[['patient_id', 'name']], on='patient_id', how='left')
threshold = patient_claims['total_billed'].quantile(0.99)  # Top 1%
high_billed_patients = patient_claims[patient_claims['total_billed'] > threshold]

    # Find overpriced procedures
procedure_costs = claims.groupby('procedure_id').agg(
    median_billed=('billed_amount', 'median'),
    mean_billed=('billed_amount', 'mean'),
    max_billed=('billed_amount', 'max')
).reset_index()

procedure_costs = procedure_costs.merge(procedures[['procedure_id', 'description']], on='procedure_id', how='left')
unusual_procedures = procedure_costs[procedure_costs['max_billed'] > (1.5 * procedure_costs['median_billed'])]


    # Identify top overcharging providers
provider_costs = claims.merge(providers, on="provider_id", how="left")

median_costs = claims.groupby("procedure_id")["billed_amount"].median().reset_index()
median_costs.rename(columns={"billed_amount": "median_cost"}, inplace=True)

provider_costs = provider_costs.merge(median_costs, on="procedure_id", how="left")
provider_costs["overcharge_ratio"] = provider_costs["billed_amount"] / provider_costs["median_cost"]

overcharging_providers = provider_costs[provider_costs["overcharge_ratio"] > 1.5]
top_overchargers = overcharging_providers.groupby(["name", "specialty"])["billed_amount"].sum().reset_index()

    # Identifying specialties with highest billing
claims_specialty = claims.merge(providers, on="provider_id", how="left")
specialty_costs = claims_specialty.groupby("specialty")["billed_amount"].sum().reset_index()

    # Find Procedures with high cost variability
procedure_variation = claims.merge(procedures, on="procedure_id", how="left").groupby("description")[
    "billed_amount"].agg(["mean", "median", "std"]).reset_index()
high_variation_procedures = procedure_variation[procedure_variation["mean"] > (1.3 * procedure_variation["median"])]

    # DENIAL RATE ANALYSIS
    # Denial rates by provider
provider_denials = claims.groupby('provider_id').agg(
    total_claims=('claim_id', 'count'),
    denied_claims=('claim_status', lambda x: (x == "Denied").sum())
).reset_index()

provider_denials['denial_rate'] = (provider_denials['denied_claims'] / provider_denials['total_claims']) * 100
provider_denials = provider_denials.merge(providers[['provider_id', 'name']], on='provider_id', how='left')

    # Denial rates by procedure
procedure_denials = claims.groupby('procedure_id').agg(
    total_claims=('claim_id', 'count'),
    denied_claims=('claim_status', lambda x: (x == "Denied").sum())
).reset_index()

procedure_denials['denial_rate'] = (procedure_denials['denied_claims'] / procedure_denials['total_claims']) * 100
procedure_denials = procedure_denials.merge(procedures[['procedure_id', 'description']], on='procedure_id', how='left')

    # Find top high billed patients
high_billed_patients = patient_claims.sort_values(by="total_billed", ascending=False).head(10)

threshold = patient_claims['total_billed'].quantile(0.99)  # Top 1%
high_billed_patients = patient_claims[patient_claims['total_billed'] > threshold]


# DISPLAY RESULTS

tools.display_dataframe_to_user(name="High Denial Rate Providers", dataframe=high_denial_providers)
tools.display_dataframe_to_user(name="Patients with High Claims", dataframe=high_billed_patients)
tools.display_dataframe_to_user(name="Unusual Procedures", dataframe=unusual_procedures)

tools.display_dataframe_to_user(name="Overcharging Providers", dataframe=top_overchargers)
tools.display_dataframe_to_user(name="High Billing Specialties", dataframe=specialty_costs)
tools.display_dataframe_to_user(name="High Cost Variability Procedures", dataframe=high_variation_procedures)

tools.display_dataframe_to_user(name="Denial Rates by Provider", dataframe=provider_denials)
tools.display_dataframe_to_user(name="Denial Rates by Procedure", dataframe=procedure_denials)

tools.display_dataframe_to_user(name="High-Billed Patients", dataframe=high_billed_patients)

print(f"High-Billed Patient Threshold: ${threshold:,.2f}")
