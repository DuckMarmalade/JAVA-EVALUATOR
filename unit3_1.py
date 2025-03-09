from langchain_groq import ChatGroq
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import ast
import re
import traceback
from unit1 import JavaSyntaxFixer
from unit2 import JavaCodeExtractor as JsonExtractor

class JavaCodeAnalyzer:
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.2):
        self.chatgroq = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )
        self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.memory_file = "java_analysis_memory.pkl"
        self.memory = self._load_memory()

        self.syntax_fixer = JavaSyntaxFixer(model_name, api_key)
        self.json_extractor = JsonExtractor(model_name, api_key)
        
        # Store all components for cross-referencing
        self.all_components = {}

    def _load_memory(self):
        """Load or initialize the memory dictionary with all required keys."""
        default_memory = {
            'embeddings': [], 
            'texts': [], 
            'component_types': [],
            'component_names': [],
            'analysis_types': []  # "first_pass", "second_pass", "evaluation"
        }
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    loaded_memory = pickle.load(f)
                    
                # Ensure all required keys exist in the loaded memory
                for key in default_memory:
                    if key not in loaded_memory:
                        loaded_memory[key] = []
                        
                # If the loaded memory has different lengths of arrays, reset it
                lengths = [len(loaded_memory[key]) for key in default_memory]
                if len(set(lengths)) > 1:  # If lengths aren't all the same
                    print("Warning: Memory file has inconsistent data lengths. Resetting memory.")
                    return default_memory
                    
                return loaded_memory
            except Exception as e:
                print(f"Error loading memory file: {str(e)}. Creating new memory.")
                return default_memory
                
        return default_memory

    def _save_memory(self):
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)

    def update_memory(self, component_type: str, component: dict, analysis: str, analysis_type: str = "second_pass"):
        """Store component analysis in vector memory with improved metadata."""
        try:
            # Get component name based on type
            component_name = ''
            if component_type == 'classes' and 'class_name' in component:
                component_name = component['class_name']
            elif component_type == 'methods' and 'method_name' in component:
                component_name = component['method_name']
            elif component_type == 'interfaces' and 'interface_name' in component:
                component_name = component['interface_name']
            elif component_type == 'constructors' and 'constructor_name' in component:
                component_name = component['constructor_name']
            elif component_type == 'fields' and 'field_name' in component:
                component_name = component['field_name']
            else:
                component_name = 'Unknown'
            
            # Create a rich context string
            context = f"""
            Type: {component_type}
            Name: {component_name}
            Analysis: {analysis}
            Description: {component.get('description', '')}
            """
            
            # Generate embedding
            embedding = self.embedding_model.encode(context)
            
            # Store in memory with enhanced metadata
            self.memory['embeddings'].append(embedding.tolist())
            self.memory['texts'].append(context)
            self.memory['component_types'].append(component_type)
            self.memory['component_names'].append(component_name)
            self.memory['analysis_types'].append(analysis_type)
            
            # Save to file
            self._save_memory()
        except Exception as e:
            print(f"Warning: Failed to update memory: {str(e)}")

    def cosine_similarity(self, a, b):
        try:
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except Exception as e:
            print(f"Warning: Similarity calculation failed: {str(e)}")
            return 0.0

    def search_similar_analyses(self, query: str, component_type: str = None, top_k: int = 3, analysis_type: str = None):
        """
        Search for similar previous analyses with filtering by component type and analysis type.
        """
        query_embedding = self.embedding_model.encode(query)
        similarities = []
        
        # Ensure memory has all required keys
        required_keys = ['embeddings', 'texts', 'component_types', 'component_names', 'analysis_types']
        for key in required_keys:
            if key not in self.memory:
                self.memory[key] = []
                
        # Return empty list if no embeddings
        if not self.memory['embeddings']:
            return []
        
        # Check if all arrays have the same length
        lengths = [len(self.memory[key]) for key in required_keys]
        if len(set(lengths)) > 1:
            print("Warning: Memory data has inconsistent lengths. Returning empty results.")
            return []
            
        for idx, stored_embedding in enumerate(self.memory['embeddings']):
            # Skip if index is out of bounds for any array
            if (idx >= len(self.memory['component_types']) or 
                idx >= len(self.memory['analysis_types']) or
                idx >= len(self.memory['texts'])):
                continue
                
            # Apply filters if specified
            if component_type and self.memory['component_types'][idx] != component_type:
                continue
                
            if analysis_type and self.memory['analysis_types'][idx] != analysis_type:
                continue
                
            score = self.cosine_similarity(query_embedding, stored_embedding)
            
            # Safe access to component name
            component_name = (
                self.memory['component_names'][idx] 
                if idx < len(self.memory['component_names']) 
                else "Unknown"
            )
            
            similarities.append((
                score, 
                self.memory['texts'][idx], 
                self.memory['component_types'][idx],
                component_name
            ))
        
        similarities.sort(reverse=True)
        return similarities[:top_k]

    def get_related_components(self, component: dict, component_type: str):
        """
        Find related components based on dependencies.
        Returns a dict of related components grouped by relationship type.
        """
        related = {}
        
        # Skip if no dependencies
        if 'dependencies' not in component or not component['dependencies']:
            return related
            
        dependencies = component['dependencies']
        
        if component_type == "classes":
            # Find super classes
            if 'super_class' in component and component['super_class']:
                related['super_classes'] = self._find_components('classes', component['super_class'])
                
            # Find implemented interfaces
            if 'implemented_interfaces' in component and component['implemented_interfaces']:
                related['interfaces'] = self._find_components('interfaces', component['implemented_interfaces'])
                
            # Find class methods
            if 'methods' in component and component['methods']:
                related['methods'] = self._find_components('methods', component['methods'])
                
        elif component_type == "methods":
            # Find called methods
            if 'called_methods' in dependencies and dependencies['called_methods']:
                related['called_methods'] = self._find_components('methods', dependencies['called_methods'])
                
            # Find used fields
            if 'used_fields' in dependencies and dependencies['used_fields']:
                related['used_fields'] = self._find_components('fields', dependencies['used_fields'])
                
            # Find method's class
            if 'enclosing_class' in component and component['enclosing_class']:
                related['enclosing_class'] = self._find_components('classes', component['enclosing_class'])
                
        # Similar patterns for other component types...
                
        return related
    
    def _find_components(self, component_type: str, names: list):
        """Helper function to find components by name in the stored components."""
        result = []
        
        if not names or component_type not in self.all_components:
            return result
            
        for name in names:
            for component in self.all_components[component_type]:
                component_name = component.get(
                    'class_name' if component_type == 'classes' 
                    else 'method_name' if component_type == 'methods' 
                    else 'interface_name' if component_type == 'interfaces' 
                    else 'constructor_name' if component_type == 'constructors'
                    else 'field_name', 
                    ''
                )
                
                if component_name == name:
                    result.append(component)
                    
        return result

    def first_pass_analyze_component(self, component: dict, component_type: str):
        """
        First pass analysis focusing on individual component properties.
        """
        details = self.prepare_analysis_prompt(component, component_type)
        
        # Search for similar analyses from first pass only (with safe implementation)
        query = f"{component_type} {details['name']} {details['description']}"
        similar_analyses = self.search_similar_analyses(
            query, 
            component_type=component_type, 
            analysis_type="first_pass"
        )
        
        # Safely handle potentially empty or differently structured similar_analyses
        previous_context = ""
        if similar_analyses:
            try:
                previous_context = "\n".join([text for _, text, _, _ in similar_analyses])
            except (ValueError, TypeError):
                # Fallback for different structure
                previous_context = "\n".join([str(item) for item in similar_analyses])
    
        prompt = f"""
        Analyze this Java {component_type} component:

        Component Name: {details['name']}
        
        Component Details:
        {details['component_details']}
        
        Implementation:
        {details['implementation']}

        {previous_context if previous_context else ""}
        
        Provide a technical analysis that includes:
        1. The purpose and functionality of this component
        2. Key characteristics (e.g., visibility, mutability, complexity)
        3. Any potential issues or concerns at the component level
        4. Code quality assessment (readability, structure, commenting)
        
        Keep your response under 100 words and focus on this component in isolation.
        """
        
        response = self.chatgroq.invoke(prompt)
        try:
            analysis = response.content.strip()
        except Exception:
            analysis = "Analysis could not be generated."
        
        component["first_pass_analysis"] = analysis
        self.update_memory(component_type, component, analysis, "first_pass")
        return component
    
    def second_pass_analyze_component(self, component: dict, component_type: str):
        """
        Second pass analysis focusing on component relationships and cross-references.
        """
        details = self.prepare_analysis_prompt(component, component_type)
        
        # Get related components
        related_components = self.get_related_components(component, component_type)
        
        # Build related components context
        related_context = ""
        for relation_type, components in related_components.items():
            related_context += f"\n{relation_type.upper()}:\n"
            for related_comp in components:
                try:
                    comp_type = self._get_component_type(related_comp)
                    comp_name = self._get_component_name(related_comp, comp_type)
                    analysis = related_comp.get("first_pass_analysis", "No analysis available")
                    related_context += f"- {comp_name} ({comp_type}): {analysis}\n"
                except Exception as e:
                    related_context += f"- Error retrieving component: {str(e)}\n"
        
        # Search for similar analyses from second pass only (with safe implementation)
        query = f"{component_type} {details['name']} {details['description']}"
        similar_analyses = self.search_similar_analyses(
            query, 
            component_type=component_type, 
            analysis_type="second_pass"
        )
        
        # Safely handle potentially empty or differently structured similar_analyses
        previous_context = ""
        if similar_analyses:
            try:
                previous_context = "\n".join([text for _, text, _, _ in similar_analyses])
            except (ValueError, TypeError):
                # Fallback for different structure
                previous_context = "\n".join([str(item) for item in similar_analyses])
        
        prompt = f"""
        DETAILED ANALYSIS FOR: {details['name']} ({component_type})

        Initial Analysis:
        {component.get('first_pass_analysis', 'Not available')}
        
        Component Details:
        {details['component_details']}
        
        Dependencies:
        {self._format_dependencies(component.get('dependencies', {}))}
        
        Related Components:
        {related_context if related_context else "No related components found."}

        {previous_context if previous_context else ""}
        
        Implementation:
        {details['implementation']}
        
        Provide a comprehensive evaluation that includes:
        1. How this component interacts with related components
        2. Dependency analysis and potential coupling issues
        3. Potential bottlenecks, inefficiencies, or design flaws
        4. Suggestions for improvement based on best practices
        5. Evaluation of error handling and edge cases
        6. Inconsistencies with related components
        
        Be specific and provide actionable insights. Focus on the component's role within the larger system.
        """
        
        response = self.chatgroq.invoke(prompt)
        try:
            analysis = response.content.strip()
        except Exception:
            analysis = "Analysis could not be generated."
        
        component["second_pass_analysis"] = analysis
        self.update_memory(component_type, component, analysis, "second_pass")
        return component
    
    def evaluate_component(self, component: dict, component_type: str):
        """
        Final evaluation focusing on scoring, best practices, and preparing for comparison.
        """
        details = self.prepare_analysis_prompt(component, component_type)
        
        first_pass = component.get('first_pass_analysis', 'Not available')
        second_pass = component.get('second_pass_analysis', 'Not available')
        
        # Safely retrieve similar analyses for evaluation
        query = f"{component_type} {details['name']} evaluation"
        similar_analyses = self.search_similar_analyses(
            query, 
            component_type=component_type, 
            analysis_type="evaluation"
        )
        
        # Safely extract previous context
        previous_context = ""
        if similar_analyses:
            try:
                previous_context = "\n".join([text for _, text, _, _ in similar_analyses])
            except (ValueError, TypeError):
                # Fallback for different structure
                previous_context = "\n".join([str(item) for item in similar_analyses])
        
        prompt = f"""
        FINAL EVALUATION FOR: {details['name']} ({component_type})

        First Pass Analysis:
        {first_pass}
        
        Second Pass Analysis:
        {second_pass}
        
        Component Details:
        {details['component_details']}
        
        Implementation:
        {details['implementation']}
        
        {previous_context if previous_context else ""}
        
        Using a structured evaluation approach, provide:
        
        1. STRENGTHS (3-5 bullet points):
           - What this component does well
           - Good practices implemented
           - Effective design choices
        
        2. WEAKNESSES (3-5 bullet points):
           - Potential issues or bugs
           - Design flaws or anti-patterns
           - Missing safeguards or validations
        
        3. RECOMMENDATIONS (3-5 bullet points):
           - Specific improvements with examples
           - Alternative approaches
           - Best practices to implement
        
        4. COMPONENT SCORE (1-10):
           - Score this component on quality, readability, efficiency, and maintainability
           - Briefly justify the score
        
        5. KEY SEMANTIC CHARACTERISTICS:
           - Identify 3-5 core semantic aspects of this component for later comparison
        """
        
        response = self.chatgroq.invoke(prompt)
        try:
            evaluation = response.content.strip()
        except Exception:
            evaluation = "Evaluation could not be generated."
        
        component["evaluation"] = evaluation
        self.update_memory(component_type, component, evaluation, "evaluation")
        return component

    def _get_component_type(self, component: dict) -> str:
        """Determine the type of a component based on its keys."""
        if 'class_name' in component:
            return 'classes'
        elif 'method_name' in component:
            return 'methods'
        elif 'interface_name' in component:
            return 'interfaces'
        elif 'constructor_name' in component:
            return 'constructors'
        elif 'field_name' in component:
            return 'fields'
        else:
            return 'unknown'
            
    def _get_component_name(self, component: dict, component_type: str) -> str:
        """Get the name of a component based on its type."""
        if not isinstance(component, dict):
            return 'Unknown'
            
        if component_type == 'classes' and 'class_name' in component:
            return component['class_name']
        elif component_type == 'methods' and 'method_name' in component:
            return component['method_name']
        elif component_type == 'interfaces' and 'interface_name' in component:
            return component['interface_name']
        elif component_type == 'constructors' and 'constructor_name' in component:
            return component['constructor_name']
        elif component_type == 'fields' and 'field_name' in component:
            return component['field_name']
        else:
            return 'Unknown'
            
    def _format_dependencies(self, dependencies: dict) -> str:
        """Format dependencies into a readable string."""
        if not dependencies:
            return "No dependencies."
            
        result = []
        for key, value in dependencies.items():
            if isinstance(value, list):
                if value:
                    formatted_value = ", ".join(str(v) for v in value)
                    result.append(f"{key}: {formatted_value}")
            elif isinstance(value, dict):
                result.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                result.append(f"{key}: {value}")
                
        return "\n".join(result) if result else "No dependencies."

    def prepare_analysis_prompt(self, component: dict, component_type: str):
        """Prepares the structured analysis details for a Java component."""
        details = {}
        
        # Helper function to safely join values
        def safe_join(values):
            if isinstance(values, list):
                return ', '.join(values)
            elif isinstance(values, str):
                return values
            elif values is None:
                return 'None'
            else:
                return str(values)
        
        # Helper to safely get parameters string
        def get_params_str(params):
            if not params or not isinstance(params, list):
                return ''
            return ', '.join(f"{p.get('name', '')}: {p.get('type', '')}" for p in params)

        if component_type == "classes":
            details["name"] = component.get('class_name', 'Not specified')
            details["component_details"] = f"""
            Class Name: {component.get('class_name', 'Not specified')}
            Modifiers: {safe_join(component.get('modifiers', []))}
            Fields: {safe_join(component.get('fields', []))}
            Methods: {safe_join(component.get('methods', []))}
            Constructors: {safe_join(component.get('constructors', []))}
            Enclosing Class: {safe_join(component.get('enclosing_class', 'None'))}
            Nested Classes/Interfaces: {safe_join(component.get('nested_classes_or_interfaces', []))}
            Super Class: {safe_join(component.get('super_class', 'None'))}
            Implemented Interfaces: {safe_join(component.get('implemented_interfaces', []))}
            Sub Class: {safe_join(component.get('sub_class', []))}
            """

        elif component_type == "methods":
            details["name"] = component.get('method_name', 'Not specified')
            details["component_details"] = f"""
            Method Name: {component.get('method_name', 'Not specified')}
            Modifiers: {safe_join(component.get('modifiers', []))}
            Return Type: {component.get('return_type', 'Not specified')}
            Parameters: {get_params_str(component.get('parameters', []))}
            Throws: {safe_join(component.get('throws', []))}
            Enclosing Class: {safe_join(component.get('enclosing_class', 'None'))}
            """

        elif component_type == "interfaces":
            details["name"] = component.get('interface_name', 'Not specified')
            details["component_details"] = f"""
            Interface Name: {component.get('interface_name', 'Not specified')}
            Modifiers: {safe_join(component.get('modifiers', []))}
            Methods: {safe_join(component.get('methods', []))}
            Default Methods: {safe_join(component.get('default_methods', []))}
            Enclosing Class: {safe_join(component.get('enclosing_class', 'None'))}
            """

        elif component_type == "constructors":
            details["name"] = component.get('constructor_name', 'Not specified')
            details["component_details"] = f"""
            Constructor Name: {component.get('constructor_name', 'Not specified')}
            Modifiers: {safe_join(component.get('modifiers', []))}
            Parameters: {get_params_str(component.get('parameters', []))}
            Throws: {safe_join(component.get('throws', []))}
            Enclosing Class: {safe_join(component.get('enclosing_class', 'None'))}
            """

        elif component_type == "fields":
            details["name"] = component.get('field_name', 'Not specified')
            details["component_details"] = f"""
            Field Name: {component.get('field_name', 'Not specified')}
            Type: {component.get('type', 'Not specified')}
            Modifiers: {safe_join(component.get('modifiers', []))}
            Initial Value: {component.get('initial_value', 'Not specified')}
            Enclosing Class: {safe_join(component.get('enclosing_class', 'None'))}
            """
            
        details['dependencies'] = component.get('dependencies', 'Not provided')
        details["implementation"] = component.get('implementation', 'Not provided')
        details["description"] = component.get('description', 'Not provided')
        
        return details

    def remove_implementation(self, components_dict):
        """Removes the implementation field from all components in the dictionary."""
        for component_type in components_dict:
            if isinstance(components_dict[component_type], list):
                for component in components_dict[component_type]:
                    if isinstance(component, dict) and 'implementation' in component:
                        component.pop('implementation')
        return components_dict

    def process_and_analyze(self, java_file_path: str):
        """
        Process and analyze Java code with dependencies included in initial extraction.
        First fixes syntax errors, then extracts components, then analyzes them.
        """
        try:
            # Step 1: Fix syntax errors
            print("Fixing syntax errors...")
            self.syntax_fixer.fix_errors_with_llm(java_file_path)
            
            # Step 2: Extract components using JsonExtractor
            print("Extracting components...")
            extracted_components = self.json_extractor.extract_components(java_file_path)
            
            # Convert the extracted components to proper format if needed
            if isinstance(extracted_components, str):
                # Clean up the JSON string
                json_string = extracted_components.replace("```json", "").replace("```", "")
                try:
                    extracted_components = json.loads(json_string)
                except json.JSONDecodeError:
                    extracted_components = ast.literal_eval(json_string)

            # Initialize the analyzed components structure
            analyzed_components = {
                "classes": [],
                "interfaces": [],
                "methods": [],
                "constructors": [],
                "fields": []
            }
            
            # Store all components for cross-referencing
            self.all_components = extracted_components

            # Step 3: Analyze components
            print("Performing first pass analysis...")
            for comp_type in analyzed_components.keys():
                for comp in extracted_components.get(comp_type, []):
                    # First pass analysis
                    analyzed_comp = self.first_pass_analyze_component(comp, comp_type)
                    # Add the component with first pass analysis
                    analyzed_components[comp_type].append(analyzed_comp)
            
            # Update all_components with first pass results
            self.all_components = analyzed_components
            
            # Step 4: Second pass analysis with cross-references
            print("Performing second pass analysis with cross-referencing...")
            for comp_type in analyzed_components.keys():
                for i, comp in enumerate(analyzed_components[comp_type]):
                    # Second pass analysis using cross-references
                    analyzed_comp = self.second_pass_analyze_component(comp, comp_type)
                    # Update the component
                    analyzed_components[comp_type][i] = analyzed_comp
            
            # Step 5: Final evaluation
            print("Performing final evaluation...")
            for comp_type in analyzed_components.keys():
                for i, comp in enumerate(analyzed_components[comp_type]):
                    # Final evaluation
                    evaluated_comp = self.evaluate_component(comp, comp_type)
                    # Update the component
                    analyzed_components[comp_type][i] = evaluated_comp

            return analyzed_components

        except Exception as e:
            import traceback
            print(f"Error in process_and_analyze: {str(e)}")
            print(traceback.format_exc())
            return None
            
    def generate_summary_report(self, analyzed_components):
        """
        Generate a summary report of the analysis.
        """
        component_counts = {comp_type: len(comps) for comp_type, comps in analyzed_components.items() if comps}
        
        # Collect all individual component scores
        all_scores = []
        for comp_type, comps in analyzed_components.items():
            for comp in comps:
                if 'evaluation' in comp:
                    # Try to extract score from evaluation
                    eval_text = comp['evaluation']
                    score_match = re.search(r'COMPONENT SCORE.*?(\d+(?:\.\d+)?)', eval_text, re.DOTALL)
                    if score_match:
                        try:
                            score = float(score_match.group(1))
                            all_scores.append(score)
                        except ValueError:
                            pass
        
        # Calculate average score if available
        avg_score = sum(all_scores) / len(all_scores) if all_scores else None
        
        # Create prompt for summary
        components_list = []
        for comp_type, comps in analyzed_components.items():
            if comps:
                for comp in comps:
                    name = self._get_component_name(comp, comp_type)
                    components_list.append(f"{name} ({comp_type})")
        
        components_text = "\n".join(f"- {comp}" for comp in components_list)
        
        prompt = f"""
        Generate a comprehensive summary report for a Java codebase analysis.
        
        Components analyzed:
        {components_text}
        
        Component counts: {json.dumps(component_counts)}
        
        Overall average score: {avg_score if avg_score is not None else 'Not available'}
        
        Based on all the individual component analyses and evaluations, provide:
        
        1. OVERALL ARCHITECTURE ASSESSMENT:
           - Describe the overall architecture and design patterns
           - Evaluate the component relationships and dependencies
           - Assess the code organization and structure
        
        2. KEY STRENGTHS:
           - Identify the major strengths of the codebase
           - Highlight well-implemented patterns or techniques
        
        3. KEY WEAKNESSES:
           - Identify the major weaknesses or areas for improvement
           - Point out potential bugs, anti-patterns, or design flaws
        
        4. RECOMMENDATIONS:
           - Provide actionable recommendations for improvement
           - Suggest refactorings or architectural changes
        
        5. FINAL SCORE:
           - Provide an overall quality score (1-10) for the codebase
           - Justify the score based on the analysis
        """
        
        response = self.chatgroq.invoke(prompt)
        return response.content.strip()

if __name__ == "__main__":
    analyzer = JavaCodeAnalyzer(
        model_name="llama-3.3-70b-versatile",
        api_key="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"
    )
    java_file_path = 'code.java'
    
    # Process and analyze (this creates the PKL file)
    analyzed_components = analyzer.process_and_analyze(java_file_path)
    
    if analyzed_components:
        # Generate summary report
        summary_report = analyzer.generate_summary_report(analyzed_components)
        
        # Clean the result for output
        cleaned_result = analyzer.remove_implementation(analyzed_components)
        
        # Save to JSON file
        output = {
            "components": cleaned_result,
            "summary_report": summary_report
        }
        
        with open('analysis_output.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        # Print the result
        print("Analysis completed successfully. Results saved to 'analysis_output.json'")
        print("\nSUMMARY REPORT:")
        print(summary_report)
    else:
        print("Analysis failed to complete successfully.")