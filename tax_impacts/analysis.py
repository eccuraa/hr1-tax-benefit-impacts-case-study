"""
Core analysis functions for tax reform impact calculations.
"""

import pandas as pd
import numpy as np
from policyengine_us import Microsimulation


def calculate_stacked_household_impacts(reforms, baseline_reform, year):
    """
    Calculate tax and income changes for each household after each reform is stacked.
    
    Parameters:
    -----------
    reforms : dict
        Dictionary of reform names to Reform objects
    baseline_reform : Reform
        The baseline reform to compare against
    year : int
        Tax year to analyze
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with household impacts
    """
    
    dataset_path = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
    
    # Calculate baseline values
    print("Calculating baseline values...")
    baseline = Microsimulation(reform=baseline_reform, dataset=dataset_path)
    
    # Get household-level baseline values
    baseline_income_tax = baseline.calculate("income_tax", map_to="household", period=year).values
    baseline_net_income = baseline.calculate("household_net_income", map_to="household", period=year).values

    # Get household-level characteristics
    household_id = baseline.calculate("household_id", map_to="household", period=year).values
    state = baseline.calculate("state_code", map_to="household", period=year).values
    num_dependents = baseline.calculate("tax_unit_dependents", map_to="household", period=year).values
    married = baseline.calculate("is_married", map_to="household", period=year).values
    employment_income = baseline.calculate("employment_income", map_to="household", period=year).values
    self_employment_income = baseline.calculate("self_employment_income", map_to="household", period=year).values
    property_taxes = baseline.calculate("real_estate_taxes", map_to="household", period=year).values
    state_income_tax = baseline.calculate("state_income_tax", map_to="household", period=year).values
    tip_income = baseline.calculate("tip_income", map_to="household", period=year).values
    overtime_income = baseline.calculate("fsla_overtime_premium", map_to="household", period=year).values
    auto_loan_interest = baseline.calculate("auto_loan_interest", map_to="household", period=year).values
    household_weight = baseline.calculate("household_weight", map_to="household", period=year).values

    married = married > 0
    
    # Get person-level values
    age = baseline.calculate("age", map_to="person", period=year).values
    person_household = baseline.calculate("household_id", map_to="person", period=year).values
    is_head = baseline.calculate("is_tax_unit_head", map_to="person", period=year).values
    is_spouse = baseline.calculate("is_tax_unit_spouse", map_to="person", period=year).values

    # Create arrays to store household-level ages (same length as household arrays)
    age_head = np.zeros(len(household_id))
    age_spouse = np.zeros(len(household_id))

    # Map person ages to household level
    for i, hh_id in enumerate(household_id):
        # Find all persons in this household
        household_mask = person_household == hh_id
        
        # Get head's age
        head_mask = household_mask & is_head
        if np.any(head_mask):
            age_head[i] = age[head_mask][0]
        
        # Get spouse's age (if married)
        if married[i]:
            spouse_mask = household_mask & is_spouse
            if np.any(spouse_mask):
                age_spouse[i] = age[spouse_mask][0]

    
    # Initialize results dictionary
    results = {
        'Household ID': household_id,
        'State': state,
        'Age of Head': age_head,
        'Age of Spouse': age_spouse,
        'Number of Dependents': num_dependents,
        'Is Married': married,
        'Employment Income': employment_income,
        'Self-Employment Income': self_employment_income,
        'Property Taxes': property_taxes,
        'State Income Tax': state_income_tax,
        'Tip Income': tip_income,
        'Overtime Income': overtime_income,
        'Auto Loan Interest': auto_loan_interest,
        'Baseline Federal Tax Liability': baseline_income_tax,
        'Baseline Net Income': baseline_net_income,
        'Household Weight': household_weight,
    }
    
    # Track cumulative values
    cumulative_reform = None
    previous_income_tax = baseline_income_tax.copy()
    previous_net_income = baseline_net_income.copy()
    
    # Apply each reform sequentially
    for reform_name, reform in reforms.items():
        print(f"Processing {reform_name}...")
        
        # Stack the reform
        if cumulative_reform is None:
            cumulative_reform = reform
        else:
            cumulative_reform = (cumulative_reform, reform)
        
        # Calculate with cumulative reforms
        reformed = Microsimulation(reform=cumulative_reform, dataset=dataset_path)
        
        # Get reformed values
        reformed_income_tax = reformed.calculate("income_tax", map_to="household", period=year).values
        reformed_net_income = reformed.calculate("household_net_income", map_to="household", period=year).values
        
        # Calculate incremental changes (from previous state)
        tax_change = reformed_income_tax - previous_income_tax
        net_income_change = reformed_net_income - previous_net_income
        
        # Store results
        results[f'Federal tax liability after {reform_name}'] = tax_change
        results[f'Net income change after {reform_name}'] = net_income_change
        
        # Update previous values for next iteration
        previous_income_tax = reformed_income_tax.copy()
        previous_net_income = reformed_net_income.copy()
    
    # Add final total changes (from baseline to fully reformed)
    results[f'Total Change in Federal Tax Liability'] = previous_income_tax - baseline_income_tax
    results[f'Total Change in Net Income'] = previous_net_income - baseline_net_income
    
    # Calculate percentage changes
    # For tax: handle cases where baseline tax is zero or negative
    pct_tax_change = np.zeros_like(baseline_income_tax)
    mask = baseline_income_tax != 0
    pct_tax_change[mask] = (results[f'Total Change in Federal Tax Liability'][mask] / baseline_income_tax[mask]) * 100
    
    # For net income: handle cases where baseline net income is zero
    pct_net_income_change = np.zeros_like(baseline_net_income)
    mask = baseline_net_income != 0
    pct_net_income_change[mask] = (results[f'Total Change in Net Income'][mask] / baseline_net_income[mask]) * 100
    
    results[f'Percentage Change in Federal Tax Liability'] = pct_tax_change
    results[f'Percentage Change in Net Income'] = pct_net_income_change
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df