import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict

class DecomposedFuzzyAHP:
    def __init__(self):
        # Optimistic linguistic scale (for "How much more important is A than B?")
        self.optimistic_scale = {
            'Exactly Equal Importance (EEI)': {'O': (0.50, 0.50), 'Saaty': 1},
            'Slightly More Important (SMI)': {'O': (0.55, 0.45), 'Saaty': 2},
            'Weakly More Important (WMI)': {'O': (0.60, 0.40), 'Saaty': 3},
            'More Important (MI)': {'O': (0.65, 0.35), 'Saaty': 4},
            'Strongly More Important (SMM)': {'O': (0.70, 0.30), 'Saaty': 5},
            'Very Strongly More Important (VSI)': {'O': (0.75, 0.25), 'Saaty': 6},
            'Absolutely More Important (AMI)': {'O': (0.80, 0.20), 'Saaty': 7},
            'Perfectly More Important (PMI)': {'O': (0.85, 0.15), 'Saaty': 8},
            'Exactly More Important (EMI)': {'O': (0.90, 0.10), 'Saaty': 9}
        }
        
        # Pessimistic linguistic scale (for "How much more unimportant is B than A?")
        self.pessimistic_scale = {
            'Exactly Equal Unimportant (EEU)': {'P': (0.50, 0.50), 'Saaty': 1},
            'Slightly Equal Unimportant (SEU)': {'P': (0.45, 0.55), 'Saaty': 2},
            'Weakly More Unimportant (WMIU)': {'P': (0.40, 0.60), 'Saaty': 3},
            'More Unimportant (MIU)': {'P': (0.35, 0.65), 'Saaty': 4},
            'Strongly More Unimportant (SMU)': {'P': (0.30, 0.70), 'Saaty': 5},
            'Very Strongly More Unimportant (VSU)': {'P': (0.25, 0.75), 'Saaty': 6},
            'Absolutely More Unimportant (AMU)': {'P': (0.20, 0.80), 'Saaty': 7},
            'Perfectly More Unimportant (PMU)': {'P': (0.15, 0.85), 'Saaty': 8},
            'Exactly More Unimportant (EMU)': {'P': (0.10, 0.90), 'Saaty': 9}
        }
        
        self.scale_multiplier = 0.90  # k value from the paper

    def linguistic_to_dfn(self, optimistic_term: str, pessimistic_term: str) -> Dict:
        """Convert linguistic terms to decomposed fuzzy numbers"""
        return {
            'O': self.optimistic_scale[optimistic_term]['O'],
            'P': self.pessimistic_scale[pessimistic_term]['P']
        }

    def get_reciprocal_pessimistic_term(self, optimistic_term: str) -> str:
        """Get the corresponding pessimistic term for a given optimistic term"""
        mapping = {
            'Exactly Equal Importance (EEI)': 'Exactly Equal Unimportant (EEU)',
            'Slightly More Important (SMI)': 'Slightly Equal Unimportant (SEU)',
            'Weakly More Important (WMI)': 'Weakly More Unimportant (WMIU)',
            'More Important (MI)': 'More Unimportant (MIU)',
            'Strongly More Important (SMM)': 'Strongly More Unimportant (SMU)',
            'Very Strongly More Important (VSI)': 'Very Strongly More Unimportant (VSU)',
            'Absolutely More Important (AMI)': 'Absolutely More Unimportant (AMU)',
            'Perfectly More Important (PMI)': 'Perfectly More Unimportant (PMU)',
            'Exactly More Important (EMI)': 'Exactly More Unimportant (EMU)'
        }
        return mapping.get(optimistic_term, 'Exactly Equal Unimportant (EEU)')

    def consistency_index(self, dfn: Dict) -> float:
        """Calculate consistency index for a decomposed fuzzy number"""
        a, b = dfn['O']  # O(μ, ϑ)
        c, d = dfn['P']  # P(μ, ϑ)
        
        numerator = math.sqrt(
            (a - d)**2 + (b - c)**2 + 
            (1 - a - b)**2 + (1 - c - d)**2
        )
        
        CI = 1 - (numerator / math.sqrt(2))
        return max(0, min(1, CI))  # Ensure between 0 and 1

    def score_index(self, dfn: Dict) -> float:
        """Calculate score index for a decomposed fuzzy number"""
        a, b = dfn['O']
        c, d = dfn['P']
        CI = self.consistency_index(dfn)
        
        score_numerator = (a + b - c + d) * CI
        score = score_numerator / (2 * self.scale_multiplier)
        
        return max(0, score)

    def dwgm_operator(self, dfn_list: List[Dict], weights: List[float] = None) -> Dict:
        """Decomposed Weighted Geometric Mean operator"""
        if weights is None:
            weights = [1/len(dfn_list)] * len(dfn_list)
        
        # Calculate O components
        a_product = 1.0
        one_minus_b_product = 1.0
        
        for i, dfn in enumerate(dfn_list):
            a, b = dfn['O']
            a_product *= a ** weights[i]
            one_minus_b_product *= (1 - b) ** weights[i]
        
        a_result = a_product
        b_result = 1 - one_minus_b_product
        
        # Calculate P components
        c_numerator = 1.0
        c_denominator_sum = 0.0
        c_denominator_product = 1.0
        
        d_sum = 0.0
        
        for i, dfn in enumerate(dfn_list):
            c, d = dfn['P']
            
            # For c component
            c_numerator *= c
            c_denominator_sum += (c ** (len(dfn_list) - 1)) * weights[i] * (1 - c)
            c_denominator_product *= c
            
            # For d component
            d_sum += weights[i] * d
        
        c_denominator = c_denominator_sum + c_denominator_product
        c_result = c_numerator / c_denominator if c_denominator != 0 else 0
        
        d_result = d_sum / (1 + sum((weights[i] * d - d) / len(dfn_list) 
                                  for i, (_, d) in enumerate([dfn['P'] for dfn in dfn_list])))
        
        return {'O': (a_result, b_result), 'P': (c_result, d_result)}

    def calculate_eigenvector(self, comparison_matrix: List[List[Dict]]) -> List[float]:
        """Calculate eigenvector weights from comparison matrix"""
        n = len(comparison_matrix)
        
        # Step 1: Aggregate each column using DWGM
        aggregated_dfns = []
        weights = [1/n] * n
        
        for j in range(n):
            column_dfns = [comparison_matrix[i][j] for i in range(n)]
            aggregated_dfns.append(self.dwgm_operator(column_dfns, weights))
        
        # Step 2: Calculate score indices for aggregated DFNs
        scores = [self.score_index(dfn) for dfn in aggregated_dfns]
        
        # Step 3: Normalize scores to get weights
        total_score = sum(scores)
        if total_score > 0:
            weights = [score / total_score for score in scores]
        else:
            weights = [1/n] * n
        
        return weights

    def calculate_consistency_ratio(self, comparison_matrix: List[List[Dict]], weights: List[float]) -> float:
        """Calculate consistency ratio (simplified version)"""
        n = len(comparison_matrix)
        
        # Convert DF-AHP matrix to Saaty-like matrix for consistency check
        saaty_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                else:
                    # Use the optimistic Saaty value
                    opt_term = None
                    for term, values in self.optimistic_scale.items():
                        if values['O'] == comparison_matrix[i][j]['O']:
                            opt_term = term
                            break
                    if opt_term:
                        row.append(self.optimistic_scale[opt_term]['Saaty'])
                    else:
                        row.append(1.0)
            saaty_matrix.append(row)
        
        # Calculate consistency (simplified)
        lambda_max = 0
        for i in range(n):
            row_sum = 0
            for j in range(n):
                row_sum += saaty_matrix[i][j] * weights[j]
            lambda_max += row_sum / weights[i]
        
        lambda_max = lambda_max / n
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        
        # Random Index values (for n up to 10)
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        
        CR = CI / RI.get(n, 1.0) if RI.get(n, 1.0) > 0 else 0
        return CR

def main():
    st.set_page_config(page_title="DF-AHP Decision Support System", layout="wide")
    
    st.title("🧠 Decomposed Fuzzy AHP (DF-AHP) Decision Support System")
    st.markdown("""
    This application implements the Decomposed Fuzzy Analytical Hierarchy Process based on the research paper:
    *"Consideration of reciprocal judgments through Decomposed Fuzzy Analytical Hierarchy Process"*
    """)
    
    # Initialize DF-AHP processor
    df_ahp = DecomposedFuzzyAHP()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Introduction", "Criteria Setup", "Pairwise Comparisons", "Results"]
    )
    
    if app_mode == "Introduction":
        st.header("Introduction to DF-AHP")
        st.markdown("""
        ### What is DF-AHP?
        The Decomposed Fuzzy Analytical Hierarchy Process (DF-AHP) is an extension of the traditional AHP method 
        that incorporates decomposed fuzzy sets to handle uncertainty and imprecision in decision-making.
        
        ### Key Features:
        - **Optimistic and Pessimistic Views**: Considers both positive and negative perspectives
        - **Reciprocal Judgments**: Automatically handles reciprocal questions like traditional AHP
        - **Fuzzy Logic**: Manages uncertainty in expert judgments
        - **Consistency Measurement**: Evaluates decision maker consistency
        
        ### How it works:
        1. Define your criteria
        2. For each pair, provide optimistic judgment (how much more important A is than B)
        3. Pessimistic judgment is automatically calculated (how much more unimportant B is than A)
        4. Get weighted results with consistency check
        """)
        
        # Display linguistic scales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimistic Scale")
            opt_data = []
            for term, values in df_ahp.optimistic_scale.items():
                opt_data.append({
                    'Term': term,
                    'O(μ,ϑ)': f"({values['O'][0]}, {values['O'][1]})",
                    'Saaty Scale': values['Saaty']
                })
            st.dataframe(pd.DataFrame(opt_data))
        
        with col2:
            st.subheader("Pessimistic Scale")
            pes_data = []
            for term, values in df_ahp.pessimistic_scale.items():
                pes_data.append({
                    'Term': term,
                    'P(μ,ϑ)': f"({values['P'][0]}, {values['P'][1]})",
                    'Saaty Scale': values['Saaty']
                })
            st.dataframe(pd.DataFrame(pes_data))
    
    elif app_mode == "Criteria Setup":
        st.header("Define Criteria")
        
        st.subheader("Main Criteria")
        num_criteria = st.number_input(
            "Number of criteria", 
            min_value=2, max_value=10, value=4
        )
        
        criteria = []
        for i in range(num_criteria):
            criterion = st.text_input(
                f"Criterion {i+1}", 
                value=f"Criterion {i+1}"
            )
            if criterion:
                criteria.append(criterion)
        
        # Store in session state
        if st.button("Save Criteria"):
            if len(criteria) >= 2:
                st.session_state.criteria = criteria
                st.success(f"Criteria saved successfully! {len(criteria)} criteria defined.")
            else:
                st.error("Please define at least 2 criteria.")
    
    elif app_mode == "Pairwise Comparisons":
        st.header("Pairwise Comparisons")
        
        if 'criteria' not in st.session_state:
            st.warning("Please set up your criteria first in the 'Criteria Setup' section.")
            return
        
        criteria = st.session_state.criteria
        n = len(criteria)
        
        st.markdown("""
        ### Instructions:
        - For each pair of criteria, select how much more important the row criterion is compared to the column criterion
        - The pessimistic judgment (reciprocal) will be automatically calculated
        - Only provide judgments for the upper triangular part (above diagonal)
        """)
        
        # Initialize comparison matrix in session state if not exists
        if 'comparison_matrix' not in st.session_state:
            st.session_state.comparison_matrix = [[None for _ in range(n)] for _ in range(n)]
        
        optimistic_terms = list(df_ahp.optimistic_scale.keys())
        
        # Create pairwise comparison matrix input
        st.subheader("Pairwise Comparison Matrix")
        
        # Display criteria labels
        cols = st.columns(n + 1)
        cols[0].write("**Criteria**")
        for j, crit in enumerate(criteria):
            cols[j + 1].write(f"**{crit}**")
        
        # Matrix input
        comparison_matrix = [[None for _ in range(n)] for _ in range(n)]
        
        for i, crit_i in enumerate(criteria):
            cols = st.columns(n + 1)
            cols[0].write(f"**{crit_i}**")
            
            for j, crit_j in enumerate(criteria):
                if i == j:
                    # Diagonal - exactly equal
                    comparison_matrix[i][j] = df_ahp.linguistic_to_dfn(
                        'Exactly Equal Importance (EEI)', 
                        'Exactly Equal Unimportant (EEU)'
                    )
                    cols[j + 1].write("EEI/EEU")
                elif i < j:
                    # Upper triangular - user input
                    selected_term = cols[j + 1].selectbox(
                        f"{crit_i} vs {crit_j}",
                        optimistic_terms,
                        index=0,
                        key=f"comp_{i}_{j}",
                        label_visibility="collapsed"
                    )
                    
                    # Get corresponding pessimistic term
                    pessimistic_term = df_ahp.get_reciprocal_pessimistic_term(selected_term)
                    
                    comparison_matrix[i][j] = df_ahp.linguistic_to_dfn(
                        selected_term, 
                        pessimistic_term
                    )
                    
                    # Display the selected terms
                    short_opt = selected_term.split('(')[1].split(')')[0]
                    short_pes = pessimistic_term.split('(')[1].split(')')[0]
                    cols[j + 1].write(f"{short_opt}/{short_pes}")
                else:
                    # Lower triangular - reciprocal (automatically filled)
                    reciprocal_optimistic = comparison_matrix[j][i]['O']
                    reciprocal_pessimistic = comparison_matrix[j][i]['P']
                    
                    # Find the terms for display
                    opt_term_display = "EEI"
                    for term, values in df_ahp.optimistic_scale.items():
                        if values['O'] == reciprocal_optimistic:
                            opt_term_display = term.split('(')[1].split(')')[0]
                            break
                    
                    pes_term_display = "EEU"
                    for term, values in df_ahp.pessimistic_scale.items():
                        if values['P'] == reciprocal_pessimistic:
                            pes_term_display = term.split('(')[1].split(')')[0]
                            break
                    
                    comparison_matrix[i][j] = {
                        'O': reciprocal_pessimistic,  # Swap for reciprocal
                        'P': reciprocal_optimistic   # Swap for reciprocal
                    }
                    cols[j + 1].write(f"{opt_term_display}/{pes_term_display}")
        
        if st.button("Calculate Weights"):
            st.session_state.comparison_matrix = comparison_matrix
            st.success("Comparisons saved! Proceed to Results.")
    
    elif app_mode == "Results":
        st.header("DF-AHP Results")
        
        if 'comparison_matrix' not in st.session_state:
            st.warning("Please complete the pairwise comparisons first.")
            return
        
        criteria = st.session_state.criteria
        comparison_matrix = st.session_state.comparison_matrix
        
        # Calculate weights
        weights = df_ahp.calculate_eigenvector(comparison_matrix)
        
        # Calculate consistency ratio
        cr = df_ahp.calculate_consistency_ratio(comparison_matrix, weights)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Criteria Weights")
            results = []
            for crit, weight in zip(criteria, weights):
                results.append({
                    'Criterion': crit,
                    'Weight': f"{weight:.4f}",
                    'Percentage': f"{(weight * 100):.2f}%"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Consistency information
            st.subheader("Consistency Check")
            st.write(f"Consistency Ratio (CR): {cr:.4f}")
            if cr <= 0.1:
                st.success("✓ Consistency is acceptable (CR ≤ 0.1)")
            else:
                st.warning("⚠ Consistency ratio is high. Consider revising judgments.")
        
        with col2:
            st.subheader("Visualization")
            
            # Bar chart
            chart_data = pd.DataFrame({
                'Criteria': criteria,
                'Weights': weights
            })
            st.bar_chart(chart_data.set_index('Criteria'))
            
            # Pie chart
            fig_data = chart_data.set_index('Criteria')
            st.write("Weight Distribution:")
            st.dataframe(fig_data)
        
        # Ranking
        st.subheader("Final Ranking")
        ranking_data = []
        for i, (crit, weight) in enumerate(zip(criteria, weights)):
            ranking_data.append({
                'Rank': i + 1,
                'Criterion': crit,
                'Weight': weight,
                'Percentage': f"{(weight * 100):.2f}%"
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Weight', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df['Weight'] = ranking_df['Weight'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(ranking_df[['Rank', 'Criterion', 'Weight', 'Percentage']])
        
        # Download results
        csv = ranking_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="df_ahp_results.csv",
            mime="text/csv"
        )
        
        # Show detailed matrix (optional)
        with st.expander("Show Detailed Comparison Matrix"):
            st.write("Decomposed Fuzzy Comparison Matrix:")
            matrix_display = []
            for i, crit_i in enumerate(criteria):
                row = {'Criterion': crit_i}
                for j, crit_j in enumerate(criteria):
                    dfn = comparison_matrix[i][j]
                    row[f"{crit_j}"] = f"O{dfn['O']}, P{dfn['P']}"
                matrix_display.append(row)
            
            st.dataframe(pd.DataFrame(matrix_display))

if __name__ == "__main__":
    main()
