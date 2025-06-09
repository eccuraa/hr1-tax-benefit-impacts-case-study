import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="HR1 Tax Impact Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Household Tax Income Changes.csv")

# Main app
def main():
    st.title("🏠 HR1 Tax Bill - Household Impact Dashboard")
    st.markdown("*Explore how the HR1 tax bill affects individual American households*")
    
    # Load the data
    df = load_data()
    
    # Sidebar for household selection
    st.sidebar.header("Select Household")
    
    # Option to select by household ID or find interesting cases
    selection_method = st.sidebar.radio(
        "Selection Method:",
        ["By Household ID", "Find Interesting Cases"]
    )
    
    if selection_method == "By Household ID":
        household_id = st.sidebar.selectbox(
            "Choose Household ID:",
            df['Household ID'].unique()
        )
    else:
        # Pre-filter for interesting cases
        interesting_options = {
            "Biggest Tax Increase": df.loc[df['Total Change in Federal Tax Liability'].idxmax(), 'Household ID'],
            "Biggest Tax Decrease": df.loc[df['Total Change in Federal Tax Liability'].idxmin(), 'Household ID'],
            "Highest Income Impact": df.loc[df['Total Change in Net Income'].abs().idxmax(), 'Household ID'],
            "Largest Percentage Change": df.loc[df['Percentage Change in Federal Tax Liability'].abs().idxmax(), 'Household ID']
        }
        
        case_type = st.sidebar.selectbox("Select Case Type:", list(interesting_options.keys()))
        household_id = interesting_options[case_type]
        st.sidebar.info(f"Selected Household ID: {household_id}")
    
    # Get household data
    household = df[df['Household ID'] == household_id].iloc[0]
    
    # Display household information in cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Baseline Attributes Card
        st.subheader("📊 Baseline Household Attributes")
        with st.container():
            st.markdown(f"""
            **Household ID:** {household['Household ID']}  
            **State:** {household['State']}  
            **Head of Household Age:** {household['Age of Head']:.0f} years  
            **Spouse Age:** {household['Age of Spouse']:.0f} years (if married)  
            **Number of Dependents:** {household['Number of Dependents']:.0f}  
            **Marital Status:** {'Married' if household['Is Married'] else 'Single'}  
            """)
            
            st.markdown("**Income Sources:**")
            income_sources = [
                ("Employment Income", household['Employment Income']),
                ("Self-Employment Income", household['Self-Employment Income']),
                ("Tip Income", household['Tip Income']),
                ("Overtime Income", household['Overtime Income'])
            ]
            
            for source, amount in income_sources:
                if amount > 0:
                    st.markdown(f"• {source}: ${amount:,.2f}")
        
        # Baseline Calculated Values Card
        st.subheader("💰 Current Tax Situation")
        with st.container():
            st.metric(
                "Federal Tax Liability", 
                f"${household['Baseline Federal Tax Liability']:,.2f}"
            )
            st.metric(
                "Net Income", 
                f"${household['Baseline Net Income']:,.2f}"
            )
            
            # Show other current expenses
            if household['Property Taxes'] > 0:
                st.markdown(f"**Property Taxes:** ${household['Property Taxes']:,.2f}")
            if household['State Income Tax'] > 0:
                st.markdown(f"**State Income Tax:** ${household['State Income Tax']:,.2f}")
    
    with col2:
        # Reform Impact Card
        st.subheader("🔄 HR1 Bill Impact Summary")
        with st.container():
            tax_change = household['Total Change in Federal Tax Liability']
            income_change = household['Total Change in Net Income']
            tax_pct_change = household['Percentage Change in Federal Tax Liability']
            income_pct_change = household['Percentage Change in Net Income']
            
            # Color coding for positive/negative changes
            tax_color = "red" if tax_change > 0 else "green"
            income_color = "green" if income_change > 0 else "red"
            
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
            <h4>Overall Impact</h4>
            <p style="color: {tax_color}; font-size: 18px; font-weight: bold;">
            Tax Change: ${tax_change:,.2f} ({tax_pct_change:+.1f}%)
            </p>
            <p style="color: {income_color}; font-size: 18px; font-weight: bold;">
            Net Income Change: ${income_change:,.2f} ({income_pct_change:+.1f}%)
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistical Weight Card
        st.subheader("📈 Statistical Weight")
        with st.container():
            weight = household['Household Weight']
            st.metric("Population Weight", f"{weight:,.0f}")
            st.caption("This household represents approximately this many similar households in the U.S.")
    
    # Detailed Reform Breakdown
    st.subheader("🔍 Detailed Reform Component Analysis")
    
    # Create reform components data
    reform_components = [
        ("Tax Rate Reform", household['Federal tax liability after Tax Rate Reform'], household['Net income change after Tax Rate Reform']),
        ("Standard Deduction Reform", household['Federal tax liability after Standard Deduction Reform'], household['Net income change after Standard Deduction Reform']),
        ("Exemption Reform", household['Federal tax liability after Exemption Reform'], household['Net income change after Exemption Reform']),
        ("Child Tax Credit Reform", household['Federal tax liability after CTC Reform'], household['Net income change after CTC Reform']),
        ("QBID Reform", household['Federal tax liability after QBID Reform'], household['Net income change after QBID Reform']),
        ("Estate Tax Reform", household['Federal tax liability after Estate Tax Reform'], household['Net income change after Estate Tax Reform']),
        ("AMT Reform", household['Federal tax liability after AMT Reform'], household['Net income change after AMT Reform']),
        ("SALT Reform", household['Federal tax liability after SALT Reform'], household['Net income change after SALT Reform']),
        ("Tip Income Exemption", household['Federal tax liability after Tip Income Exempt'], household['Net income change after Tip Income Exempt']),
        ("Overtime Income Exemption", household['Federal tax liability after Overtime Income Exempt'], household['Net income change after Overtime Income Exempt']),
        ("Auto Loan Interest Deduction", household['Federal tax liability after Auto Loan Interest ALD'], household['Net income change after Auto Loan Interest ALD'])
    ]
    
    # Filter out components with no change
    active_components = [(name, tax_after, income_change) for name, tax_after, income_change in reform_components if abs(income_change) > 0.01]
    
    if active_components:
        cols = st.columns(min(3, len(active_components)))
        for i, (name, tax_after, income_change) in enumerate(active_components):
            with cols[i % 3]:
                color = "green" if income_change > 0 else "red"
                st.markdown(f"""
                <div style="padding: 8px; border-radius: 5px; background-color: #f9f9f9; margin: 5px 0;">
                <h5>{name}</h5>
                <p style="color: {color}; font-weight: bold;">
                Income Change: ${income_change:,.2f}
                </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("This household is not significantly affected by any specific reform components.")
    
    # Summary for journalists
    st.subheader("📝 Story Summary")
    impact_level = "significantly" if abs(income_change) > 1000 else "moderately" if abs(income_change) > 100 else "minimally"
    direction = "benefits from" if income_change > 0 else "is burdened by"
    
    st.info(f"""
    **Quick Story Angle:** This {household['State']} household (ID: {household_id}) {impact_level} {direction} the HR1 bill, 
    with a net income change of ${income_change:,.2f} ({income_pct_change:+.1f}%). 
    The household represents approximately {weight:,.0f} similar American families.
    """)

if __name__ == "__main__":
    main()