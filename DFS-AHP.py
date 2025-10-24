import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict

class DecomposedFuzzyAHP:
    def __init__(self):
        self.linguistic_scale = {
            # Optimistic linguistic terms
            'EEI': {'O': (0.50, 0.50), 'P': (0.50, 0.50), 'Saaty': 1},
            'SMI': {'O': (0.55, 0.45), 'P': (0.45, 0.55), 'Saaty': 2},
            'WMI': {'O': (0.60, 0.40), 'P': (0.40, 0.60), 'Saaty': 3},
            'MI': {'O': (0.65, 0.35), 'P': (0.35, 0.65), 'Saaty': 4},
            'SMM': {'O': (0.70, 0.30), 'P': (0.30, 0.70), 'Saaty': 5},
            'VSI': {'O': (0.75, 0.25), 'P': (0.25, 0.75), 'Saaty': 6},
            'AMI': {'O': (0.80, 0.20), 'P': (0.20, 0.80), 'Saaty': 7},
            'PMI': {'O': (0.85, 0.15), 'P': (0.15, 0.85), 'Saaty': 8},
            'EMI': {'O': (0.90, 0.10), 'P': (0.10, 0.90), 'Saaty': 9}
        }
        
        self.scale_multiplier = 0.90  # k value from the paper

    def linguistic_to_dfn(self, optimistic_term: str, pessimistic_term: str) -> Dict:
        """Convert linguistic terms to decomposed fuzzy numbers"""
        return {
            'O': self.linguistic_scale[optimistic_term]['O'],
            'P': self.linguistic_scale[pessimistic_term]['P']
        }

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

    def calculate_weights(self, comparison_matrices: List[List[Dict]]) -> List[float]:
        """Calculate final weights from comparison matrices"""
        aggregated_weights = []
        
        for matrix in comparison_matrices:
            # Aggregate each column using DWGM
            n = len(matrix)
            weights = [1/n] * n
            aggregated_dfns = []
            
            for j in range(n):
                column_dfns = [matrix[i][j] for i in range(n)]
                aggregated_dfns.append(self.dwgm_operator(column_dfns, weights))
            
            # Calculate score indices for aggregated DFNs
            scores = [self.score_index(dfn) for dfn in aggregated_dfns]
            
            # Normalize scores
            total_score = sum(scores)
            if total_score > 0:
                normalized_weights = [score / total_score for score in scores]
            else:
                normalized_weights = [1/len(scores)] * len(scores)
            
            aggregated_weights.append(normalized_weights)
        
        return aggregated_weights

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
        - **Reciprocal Judgments**: Handles functional and dysfunctional questions
        - **Fuzzy Logic**: Manages uncertainty in expert judgments
        - **Consistency Measurement**: Evaluates decision maker consistency
        
        ### Linguistic Scale:
        The method uses a 9-point scale ranging from "Exactly Equal Importance" to "Exactly More Important" 
        for both optimistic and pessimistic judgments.
        """)
        
        # Display linguistic scale
        scale_data = []
        for term, values in df_ahp.linguistic_scale.items():
            scale_data.append({
                'Optimistic Term': term,
                'O(μ,ϑ)': f"({values['O'][0]}, {values['O'][1]})",
                'Pessimistic Term': term.replace('I', 'U') if 'I' in term else term,
                'P(μ,ϑ)': f"({values['P'][0]}, {values['P'][1]})",
                'Saaty Scale': values['Saaty']
            })
        
        st.subheader("Linguistic Scale Table")
        st.dataframe(pd.DataFrame(scale_data))
    
    elif app_mode == "Criteria Setup":
        st.header("Define Criteria Hierarchy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Main Criteria")
            num_main_criteria = st.number_input(
                "Number of main criteria", 
                min_value=2, max_value=10, value=4
            )
            
            main_criteria = []
            for i in range(num_main_criteria):
                criterion = st.text_input(
                    f"Main criterion {i+1}", 
                    value=f"Criterion {i+1}"
                )
                main_criteria.append(criterion)
        
        with col2:
            st.subheader("Sub-criteria")
            num_sub_criteria = st.number_input(
                "Number of sub-criteria per main criterion",
                min_value=1, max_value=10, value=3
            )
            
            sub_criteria = {}
            for i, main_crit in enumerate(main_criteria):
                st.write(f"**{main_crit}**")
                sub_crit_list = []
                for j in range(num_sub_criteria):
                    sub_crit = st.text_input(
                        f"Sub-criterion {j+1} for {main_crit}",
                        value=f"{main_crit}.{j+1}",
                        key=f"sub_{i}_{j}"
                    )
                    sub_crit_list.append(sub_crit)
                sub_criteria[main_crit] = sub_crit_list
        
        # Store in session state
        st.session_state.main_criteria = main_criteria
        st.session_state.sub_criteria = sub_criteria
        
        if st.button("Save Criteria Hierarchy"):
            st.success("Criteria hierarchy saved successfully!")
            st.write("You can now proceed to pairwise comparisons.")
    
    elif app_mode == "Pairwise Comparisons":
        st.header("Pairwise Comparisons")
        
        if 'main_criteria' not in st.session_state:
            st.warning("Please set up your criteria hierarchy first in the 'Criteria Setup' section.")
            return
        
        main_criteria = st.session_state.main_criteria
        sub_criteria = st.session_state.sub_criteria
        
        st.subheader("Main Criteria Comparisons")
        st.info("For each pair, select the optimistic and pessimistic linguistic terms")
        
        # Main criteria comparison matrix
        main_comparisons = []
        linguistic_terms = list(df_ahp.linguistic_scale.keys())
        
        for i, crit1 in enumerate(main_criteria):
            row_comparisons = []
            for j, crit2 in enumerate(main_criteria):
                if i == j:
                    # Diagonal elements are exactly equal
                    dfn = df_ahp.linguistic_to_dfn('EEI', 'EEI')
                    row_comparisons.append(dfn)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{crit1}** vs **{crit2}**")
                    
                    with col2:
                        optimistic = st.selectbox(
                            f"Optimistic: {crit1} over {crit2}",
                            linguistic_terms,
                            index=0,
                            key=f"main_opt_{i}_{j}"
                        )
                        pessimistic = st.selectbox(
                            f"Pessimistic: {crit2} over {crit1}",
                            linguistic_terms,
                            index=0,
                            key=f"main_pes_{i}_{j}"
                        )
                    
                    dfn = df_ahp.linguistic_to_dfn(optimistic, pessimistic)
                    row_comparisons.append(dfn)
            
            main_comparisons.append(row_comparisons)
        
        # Sub-criteria comparisons
        st.subheader("Sub-criteria Comparisons")
        
        sub_comparisons_dict = {}
        for main_crit, sub_crit_list in sub_criteria.items():
            st.write(f"**{main_crit}** sub-criteria comparisons")
            
            sub_comparisons = []
            for i, sub1 in enumerate(sub_crit_list):
                row_sub_comparisons = []
                for j, sub2 in enumerate(sub_crit_list):
                    if i == j:
                        dfn = df_ahp.linguistic_to_dfn('EEI', 'EEI')
                        row_sub_comparisons.append(dfn)
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{sub1}** vs **{sub2}**")
                        
                        with col2:
                            optimistic = st.selectbox(
                                f"Optimistic: {sub1} over {sub2}",
                                linguistic_terms,
                                index=0,
                                key=f"sub_{main_crit}_{i}_{j}"
                            )
                            pessimistic = st.selectbox(
                                f"Pessimistic: {sub2} over {sub1}",
                                linguistic_terms,
                                index=0,
                                key=f"sub_pes_{main_crit}_{i}_{j}"
                            )
                        
                        dfn = df_ahp.linguistic_to_dfn(optimistic, pessimistic)
                        row_sub_comparisons.append(dfn)
                
                sub_comparisons.append(row_sub_comparisons)
            
            sub_comparisons_dict[main_crit] = sub_comparisons
        
        if st.button("Calculate Weights"):
            st.session_state.main_comparisons = main_comparisons
            st.session_state.sub_comparisons = sub_comparisons_dict
            st.success("Comparisons saved! Proceed to Results.")
    
    elif app_mode == "Results":
        st.header("DF-AHP Results")
        
        if 'main_comparisons' not in st.session_state:
            st.warning("Please complete the pairwise comparisons first.")
            return
        
        # Calculate weights
        df_ahp = DecomposedFuzzyAHP()
        main_comparisons = st.session_state.main_comparisons
        sub_comparisons = st.session_state.sub_comparisons
        main_criteria = st.session_state.main_criteria
        sub_criteria = st.session_state.sub_criteria
        
        # Calculate main criteria weights
        main_weights = df_ahp.calculate_weights([main_comparisons])[0]
        
        # Calculate sub-criteria weights
        sub_weights_dict = {}
        for main_crit in main_criteria:
            if main_crit in sub_comparisons:
                sub_weights = df_ahp.calculate_weights([sub_comparisons[main_crit]])[0]
                sub_weights_dict[main_crit] = sub_weights
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Main Criteria Weights")
            main_results = []
            for crit, weight in zip(main_criteria, main_weights):
                main_results.append({
                    'Criterion': crit,
                    'Weight': f"{weight:.4f}",
                    'Percentage': f"{(weight * 100):.2f}%"
                })
            
            main_df = pd.DataFrame(main_results)
            st.dataframe(main_df)
            
            # Visualization
            st.subheader("Main Criteria Distribution")
            chart_data = pd.DataFrame({
                'Criteria': main_criteria,
                'Weights': main_weights
            })
            st.bar_chart(chart_data.set_index('Criteria'))
        
        with col2:
            st.subheader("Sub-criteria Weights")
            
            for main_crit in main_criteria:
                if main_crit in sub_weights_dict:
                    st.write(f"**{main_crit}**")
                    sub_results = []
                    for sub_crit, weight in zip(sub_criteria[main_crit], sub_weights_dict[main_crit]):
                        sub_results.append({
                            'Sub-criterion': sub_crit,
                            'Weight': f"{weight:.4f}"
                        })
                    
                    sub_df = pd.DataFrame(sub_results)
                    st.dataframe(sub_df)
        
        # Final ranking
        st.subheader("Final Ranking")
        final_ranking = []
        
        for i, main_crit in enumerate(main_criteria):
            main_weight = main_weights[i]
            if main_crit in sub_weights_dict:
                for j, sub_crit in enumerate(sub_criteria[main_crit]):
                    sub_weight = sub_weights_dict[main_crit][j]
                    global_weight = main_weight * sub_weight
                    final_ranking.append({
                        'Criterion': f"{main_crit} - {sub_crit}",
                        'Global Weight': global_weight
                    })
        
        final_df = pd.DataFrame(final_ranking)
        final_df = final_df.sort_values('Global Weight', ascending=False)
        final_df['Rank'] = range(1, len(final_df) + 1)
        final_df['Global Weight'] = final_df['Global Weight'].apply(lambda x: f"{x:.6f}")
        
        st.dataframe(final_df[['Rank', 'Criterion', 'Global Weight']])
        
        # Download results
        csv = final_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="df_ahp_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
