import os
import pandas as pd
import numpy as np

def generate_samples():
    os.makedirs("backend/data/samples", exist_ok=True)
    os.makedirs("backend/data/uploads", exist_ok=True)
    os.makedirs("backend/data/artifacts", exist_ok=True)
    os.makedirs("backend/data/reports", exist_ok=True)

    np.random.seed(42)
    n = 5000

    # 1. Loan Data (Classification) matching reference UI
    ages = np.random.randint(18, 66, size=n)
    incomes = np.random.normal(50000, 21000, size=n).clip(12000, 150000).round(-2)
    credit_scores = np.random.normal(680, 62, size=n).clip(350, 850).astype(int)
    experience = (ages - np.random.randint(18, 26, size=n)).clip(0, 40)
    existing_loans = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.35, 0.35, 0.18, 0.08, 0.04])
    loan_amounts = (incomes * np.random.uniform(0.1, 0.45, size=n)).round(-2)
    
    education = np.random.choice(["High School", "Bachelor", "Master", "PhD"], size=n, p=[0.25, 0.50, 0.20, 0.05])
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    employment_type = np.random.choice(["Salaried", "Self-Employed", "Business", "Unemployed"], size=n, p=[0.60, 0.22, 0.13, 0.05])
    home_ownership = np.random.choice(["Rent", "Own", "Mortgage"], size=n, p=[0.45, 0.20, 0.35])
    has_credit_card = np.random.choice(["Yes", "No"], size=n, p=[0.78, 0.22])
    dependents = np.random.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1])
    savings_balance = np.random.exponential(15000, size=n).clip(500, 100000).round(2)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], size=n, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    id_col = [f"LOAN_{10000 + i}" for i in range(n)]

    # Probability of approval
    z = (
        (credit_scores - 650) / 70
        + (incomes - 45000) / 30000
        - (loan_amounts - 15000) / 20000
        - existing_loans * 0.4
        + (experience / 10) * 0.3
        + (employment_type == "Salaried") * 0.5
        + (education == "Master") * 0.3
        + (education == "PhD") * 0.4
        - (employment_type == "Unemployed") * 1.8
        + np.random.normal(0, 0.6, size=n)
    )
    prob = 1 / (1 + np.exp(-z))
    loan_status = np.where(prob >= 0.52, "Approved", "Rejected")

    df_loan = pd.DataFrame({
        "Customer_ID": id_col,
        "Age": ages,
        "Income": incomes,
        "Credit_Score": credit_scores,
        "Experience": experience,
        "Existing_Loans": existing_loans,
        "Loan_Amount": loan_amounts,
        "Savings_Balance": savings_balance,
        "Loan_Term_Months": loan_term_months,
        "Dependents": dependents,
        "Education": education,
        "Gender": gender,
        "Employment_Type": employment_type,
        "Home_Ownership": home_ownership,
        "Has_Credit_Card": has_credit_card,
        "Loan_Status": loan_status
    })

    # Add a few realistic missing values and duplicates matching the screenshot
    missing_indices = np.random.choice(n, size=37, replace=False)
    for idx in missing_indices[:15]:
        df_loan.loc[idx, "Income"] = np.nan
    for idx in missing_indices[15:25]:
        df_loan.loc[idx, "Savings_Balance"] = np.nan
    for idx in missing_indices[25:]:
        df_loan.loc[idx, "Employment_Type"] = np.nan

    # 4 duplicate rows
    duplicates = df_loan.iloc[:4].copy()
    df_loan = pd.concat([df_loan, duplicates], ignore_index=True)

    loan_csv_path = "backend/data/samples/loan_data.csv"
    df_loan.to_csv(loan_csv_path, index=False)
    print(f"Generated {loan_csv_path} with shape {df_loan.shape}")

    # 2. House Prices Data (Regression)
    n_house = 2000
    sqft = np.random.randint(600, 4500, size=n_house)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_house, p=[0.05, 0.25, 0.45, 0.20, 0.05])
    bathrooms = (bedrooms * 0.75 + np.random.uniform(0.5, 1.5, size=n_house)).round().clip(1, 5)
    year_built = np.random.randint(1950, 2024, size=n_house)
    garage_cars = np.random.choice([0, 1, 2, 3], size=n_house, p=[0.1, 0.3, 0.45, 0.15])
    neighborhood = np.random.choice(["Downtown", "Suburbs", "Uptown", "Rural", "Waterfront"], size=n_house, p=[0.25, 0.35, 0.2, 0.15, 0.05])
    condition_rating = np.random.randint(1, 6, size=n_house)

    base_price = (
        sqft * 160
        + bedrooms * 12000
        + bathrooms * 18000
        + (year_built - 1950) * 800
        + garage_cars * 15000
        + condition_rating * 10000
        + (neighborhood == "Waterfront") * 120000
        + (neighborhood == "Downtown") * 60000
        + np.random.normal(0, 25000, size=n_house)
    ).clip(60000, 1200000).round(-2)

    df_house = pd.DataFrame({
        "Property_ID": [f"PROP_{5000 + i}" for i in range(n_house)],
        "Square_Feet": sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Year_Built": year_built,
        "Garage_Cars": garage_cars,
        "Condition_Rating": condition_rating,
        "Neighborhood": neighborhood,
        "Sale_Price": base_price
    })

    house_csv_path = "backend/data/samples/house_prices.csv"
    df_house.to_csv(house_csv_path, index=False)
    print(f"Generated {house_csv_path} with shape {df_house.shape}")

if __name__ == "__main__":
    generate_samples()
