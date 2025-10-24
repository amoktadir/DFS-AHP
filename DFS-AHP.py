import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict

class DecomposedFuzzyAHP:
    def __init__(self):
        # Complete linguistic scale from Table 2 in the paper
        self.linguistic_scale = {
            # Format: (Optimistic Term, Pessimistic Term): (O(μ,ϑ), P(μ,ϑ))
            ('EEI', 'EEU'): {'O': (0.50, 0.50), 'P': (0.50, 0.50), 'Saaty': 1},
            ('SMI', 'SEU'): {'O': (0.55, 0.45), 'P': (0.45, 0.55), 'Saaty': 2},
            ('WMI', 'WMIU'): {'O': (0.60, 0.40), 'P': (0.40, 0.60), 'Saaty': 3},
            ('MI', 'MIU'): {'O': (0.65, 0.35), 'P': (0.35, 0.65), 'Saaty': 4},
            ('SMM', 'SMU'): {'O': (0.70, 0.30), 'P': (0.30, 0.70), 'Saaty': 5},
            ('VSI', 'VSU'): {'O': (0.75, 0.25), 'P': (0.25, 0.75), 'Saaty': 6},
            ('AMI', 'AMU'): {'O': (0.80, 0.20), 'P': (0.20, 0.80), 'Saaty': 7},
            ('PMI', 'PMU'): {'O': (0.85, 0.15), 'P': (0.15, 0.85), 'Saaty': 8},
            ('EMI', 'EMU'): {'O': (0.90, 0.10), 'P': (0.10, 0.90), 'Saaty': 9}
        }
        
        self.scale_multiplier = 0.90  # k value from the paper

    def get_dfn_by_terms(self, optimistic_term: str, pessimistic_term: str) -> Dict:
        """Get decomposed fuzzy number by optimistic and pessimistic terms"""
        key = (optimistic_term, pessimistic_term)
        if key in self.linguistic_scale:
            return {
                'O': self.linguistic_scale[key]['O'],
                'P': self.linguistic_scale[key]['P']
            }
        else:
            # Default to equal importance
            return {'O': (0.50, 0.50), 'P': (0.50, 0.50)}

    def get_reciprocal_terms(self, optimistic_term: str, pessimistic_term: str) -> tuple:
        """Get reciprocal terms for the comparison"""
        # For reciprocal comparison (B vs A), we swap the terms
        return (pessimistic_term, optimistic_term)

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
        """Calculate consistency ratio"""
        n = len(comparison_matrix)
        if n <= 1:
            return 0.0
            
        # Convert to Saaty-like matrix for consistency check
        saaty_matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dfn = comparison_matrix[i][j]
                    # Find matching linguistic terms
                    for (opt_term, pes_term), values in self.linguistic_scale.items():
                        if values['O'] == dfn['O'] and values['P'] == dfn['P']:
                            saaty_matrix[i][j] = values['Saaty']
                            break
        
        # Calculate consistency
        weighted_sum = np.dot(saaty_matrix, weights)
        lambda_max = np.sum(weighted_sum / weights) / n
        
        CI = (lambda_max - n) / (n - 1)
        
        # Random Index values
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
        - **Reciprocal Judgments**: Automatically handles reciprocal relationships
        - **Fuzzy Logic**: Manages uncertainty in expert judgments
        - **Consistency Measurement**: Evaluates decision maker consistency
        
        ### How it works:
        For each pair of criteria (A vs B), you provide:
        - **Optimistic judgment**: How much more important is A than B?
        - **Pessimistic judgment**: How much more unimportant is B than A?
        
        The system automatically handles the reciprocal comparisons (B vs A).
        """)
        
        # Display linguistic scale
        st.subheader("Linguistic Scale Table")
        scale_data = []
        for (opt_term, pes_term), values in df_ahp.linguistic_scale.items():
            scale_data.append({
                'Optimistic Term': opt_term,
                'Pessimistic Term': pes_term,
                'O(μ,ϑ)': f"({values['O'][0]}, {values['O'][1]})",
                'P(μ,ϑ)': f"({values['P'][0]}, {values['P'][1]})",
                'Saaty Scale': values['Saaty']
            })
        
        st.dataframe(pd.DataFrame(scale_data))
        
        # Show example from the paper
        st.subheader("Example from Research Paper")
        st.markdown("""
        The table below shows Expert 1's judgments from the research paper:
        """)
        
        example_data = {
            'Criteria': ['Delivery', 'Quality', 'Operation', 'Technology'],
            'Delivery': ['(EEM, EEM)', '(AMU, AMM)', '(MU, SMM)', '(WMU, WMM)'],
            'Quality': ['(AMM, AMU)', '(EEM, EEM)', '(MM, MU)', '(AMM, VSU)'],
            'Operation': ['(MM, SMU)', '(MU, MM)', '(EEM, EEM)', '(WMM, WMU)'],
            'Technology': ['(WMM, WMU)', '(AMU, AMM)', '(WMU, WMM)', '(EEM, EEM)']
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)
    
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
                
                # Initialize comparison matrix
                n = len(criteria)
                comparison_matrix = [[None for _ in range(n)] for _ in range(n)]
                
                # Set diagonal to EEM/EEM
                for i in range(n):
                    comparison_matrix[i][i] = df_ahp.get_dfn_by_terms('EEI', 'EEU')
                
                st.session_state.comparison_matrix = comparison_matrix
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
        - For each pair of criteria, provide both optimistic and pessimistic judgments
        - Format: **(Optimistic, Pessimistic)** where:
          - **Optimistic**: How much more important is the row criterion than the column criterion
          - **Pessimistic**: How much more unimportant is the column criterion than the row criterion
        - Only provide judgments for the upper triangular part (above diagonal)
        - Lower triangular will be automatically filled with reciprocals
        """)
        
        # Get all possible term combinations
        term_combinations = list(df_ahp.linguistic_scale.keys())
        
        # Initialize comparison matrix in session state if not exists
        if 'comparison_matrix' not in st.session_state:
            st.session_state.comparison_matrix = [[None for _ in range(n)] for _ in range(n)]
            # Set diagonal to EEM/EEM
            for i in range(n):
                st.session_state.comparison_matrix[i][i] = df_ahp.get_dfn_by_terms('EEI', 'EEU')
        
        comparison_matrix = st.session_state.comparison_matrix
        
        # Display comparison matrix
        st.subheader("Pairwise Comparison Matrix")
        
        # Create header row
        cols = st.columns(n + 1)
        cols[0].write("**Criteria**")
        for j, crit in enumerate(criteria):
            cols[j + 1].write(f"**{crit}**")
        
        # Matrix input
        for i, crit_i in enumerate(criteria):
            cols = st.columns(n + 1)
            cols[0].write(f"**{crit_i}**")
            
            for j, crit_j in enumerate(criteria):
                if i == j:
                    # Diagonal - display only
                    cols[j + 1].write("(EEI, EEU)")
                    comparison_matrix[i][j] = df_ahp.get_dfn_by_terms('EEI', 'EEU')
                elif i < j:
                    # Upper triangular - user input
                    current_value = comparison_matrix[i][j]
                    current_index = 0
                    if current_value:
                        # Find current selection
                        for idx, (opt, pes) in enumerate(term_combinations):
                            if (df_ahp.linguistic_scale[(opt, pes)]['O'] == current_value['O'] and 
                                df_ahp.linguistic_scale[(opt, pes)]['P'] == current_value['P']):
                                current_index = idx
                                break
                    
                    selected_option = cols[j + 1].selectbox(
                        f"Select comparison for {crit_i} vs {crit_j}",
                        [f"({opt}, {pes})" for (opt, pes) in term_combinations],
                        index=current_index,
                        key=f"comp_{i}_{j}",
                        label_visibility="collapsed"
                    )
                    
                    # Extract terms from selection
                    selected_terms = selected_option.strip('()').split(', ')
                    opt_term, pes_term = selected_terms[0], selected_terms[1]
                    
                    comparison_matrix[i][j] = df_ahp.get_dfn_by_terms(opt_term, pes_term)
                    
                    # Automatically set reciprocal (lower triangular)
                    recip_opt, recip_pes = df_ahp.get_reciprocal_terms(opt_term, pes_term)
                    comparison_matrix[j][i] = df_ahp.get_dfn_by_terms(recip_opt, recip_pes)
                else:
                    # Lower triangular - display reciprocal (automatically set)
                    if comparison_matrix[i][j]:
                        # Find the terms for display
                        display_terms = None
                        for (opt, pes), values in df_ahp.linguistic_scale.items():
                            if (values['O'] == comparison_matrix[i][j]['O'] and 
                                values['P'] == comparison_matrix[i][j]['P']):
                                display_terms = f"({opt}, {pes})"
                                break
                        if display_terms:
                            cols[j + 1].write(display_terms)
                        else:
                            cols[j + 1].write("(EEI, EEU)")
                    else:
                        cols[j + 1].write("(EEI, EEU)")
        
        if st.button("Save Comparisons and Calculate Weights"):
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
            
            # Weight distribution
            st.write("Weight Distribution:")
            fig_data = chart_data.set_index('Criteria')
            st.dataframe(fig_data)
        
        # Final ranking
        st.subheader("Final Ranking")
        ranking_data = []
        for i, (crit, weight) in enumerate(zip(criteria, weights)):
            ranking_data.append({
                'Criterion': crit,
                'Weight': weight,
                'Percentage': f"{(weight * 100):.2f}%"
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Weight', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df['Weight'] = ranking_df['Weight'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(ranking_df[['Rank', 'Criterion', 'Weight', 'Percentage']].reset_index(drop=True))
        
        # Download results
        csv = ranking_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="df_ahp_results.csv",
            mime="text/csv"
        )
        
        # Show detailed matrix
        with st.expander("Show Detailed Comparison Matrix"):
            st.write("Complete Decomposed Fuzzy Comparison Matrix:")
            matrix_display = []
            for i, crit_i in enumerate(criteria):
                row = {'Criterion': crit_i}
                for j, crit_j in enumerate(criteria):
                    dfn = comparison_matrix[i][j]
                    # Find matching terms
                    terms_display = "(EEI, EEU)"
                    for (opt, pes), values in df_ahp.linguistic_scale.items():
                        if values['O'] == dfn['O'] and values['P'] == dfn['P']:
                            terms_display = f"({opt}, {pes})"
                            break
                    row[f"{crit_j}"] = terms_display
                matrix_display.append(row)
            
            st.dataframe(pd.DataFrame(matrix_display))

if __name__ == "__main__":
    main()
