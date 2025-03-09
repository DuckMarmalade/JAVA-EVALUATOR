from langchain_groq import ChatGroq
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import ast
from unit1 import JavaSyntaxFixer
from unit2 import JavaCodeExtractor as JsonExtractor
class JavaCodeExtractor:
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

# Memory elements are stored as a dictionary with keys 'embeddings' and 'texts'

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                return pickle.load(f)
        return {'embeddings': [], 'texts': []}

    def _save_memory(self):
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)

    def update_memory(self, component_type: str, component: dict, analysis: str):
        """Store component analysis in vector memory."""
        try:
            # Create a rich context string
            context = f"""
            Type: {component_type}
            Name: {component.get('class_name' if component_type == 'classes' else 'method_name' if component_type == 'methods' else 'interface_name' if component_type == 'interfaces' else 'field_name', 'Unknown')}
            Analysis: {analysis}
            Description: {component.get('description', '')}
            """
            
            # Generate embedding
            embedding = self.embedding_model.encode(context)
            
            # Store in memory
            self.memory['embeddings'].append(embedding.tolist())
            self.memory['texts'].append(context)
            
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

    def search_similar_analyses(self, query: str, top_k: int = 3):
        """Search for similar previous analyses."""
        query_embedding = self.embedding_model.encode(query)
        similarities = []
        
        for idx, stored_embedding in enumerate(self.memory['embeddings']):
            score = self.cosine_similarity(query_embedding, stored_embedding)
            similarities.append((score, self.memory['texts'][idx]))
        
        similarities.sort(reverse=True)
        return similarities[:top_k]

    def first_pass_analyze_component(self, component: dict, component_type: str):
        """First pass."""
        details = self.prepare_analysis_prompt(component, component_type)
        # memory element components
        query = f"{component_type} {details['name']} {details['description']}"
        similar_analyses = self.search_similar_analyses(query)
        previous_context = "\n".join([text for _, text in similar_analyses])
    
        prompt = f"""
        Previous similar analyses:
        {previous_context}\n,

        {details['name']}, {component_type}: Analyze this Java component.

        Component Details:
        {details['component_details']}
        
        Implementation:
        {details['implementation']}

        Original Description:
        {details['description']}
        
        Provide a brief technical analysis in less than 50 words.
        """
        
        response = self.chatgroq.invoke(prompt)
        try:
            analysis = response.content.strip()
        except Exception:
            analysis = "Analysis could not be generated."
        
        component["first_Pass_Analysis"] = analysis
        self.update_memory(component_type, component, analysis)
        return component
    
    def second_pass_analyze_component(self, component: dict, component_type: str):
        """Second pass"""
        details = self.prepare_analysis_prompt(component, component_type)
        query = f"{component_type} {details['name']} {details['description']}"
        similar_analyses = self.search_similar_analyses(query)
        previous_context = "\n".join([text for _, text in similar_analyses])
        
        prompt = f"""
        {details['name']}, {component_type}: Give a detailed analysis of this Java component given the following.

        Previous similar analyses:
        {previous_context}
        
        Component Details:
        {details['component_details']}
        
        Implementation:
        {details['implementation']}

        Original Description:
        {details['description']}

        Dependencies:
        {details['dependencies']}

        initial_analysis:
        {details['first_Pass_Analysis']}
        
        """
        
        response = self.chatgroq.invoke(prompt)
        try:
            analysis = response.content.strip()
        except Exception:
            analysis = "Analysis could not be generated."
        
        component["second_Pass_Analysis"] = analysis
        # print(analysis)
        self.update_memory(component_type, component, analysis)
        return component



    def prepare_analysis_prompt(self, component: dict, component_type: str):
        """Prepares the structured analysis details for a Java component."""
        details = {}

        if component_type == "classes":
            details["name"]=component.get('class_name', 'Not specified')
            details["component_details"] = f"""
            Class Name: {component.get('class_name', 'Not specified')}
            Modifiers: {', '.join(component.get('modifiers', []))}
            Fields: {', '.join(component.get('fields', []))}
            Methods: {', '.join(component.get('methods', []))}
            Constructors: {', '.join(component.get('constructors', []))}
            Enclosing Class: {', '.join(component.get('enclosing_class', []))}
            Nested Classes/Interfaces: {', '.join(component.get('nested_classes_or_interfaces', []))}
            Super Class: {', '.join(component.get('super_class', []))}
            Implemented Interfaces: {', '.join(component.get('implemented_interfaces', []))}
            Sub Class: {', '.join(component.get('sub_class', []))}
            """

        elif component_type == "methods":
            details["name"]=component.get('method_name', 'Not specified')
            details["component_details"] = f"""
            Method Name: {component.get('method_name', 'Not specified')}
            Modifiers: {', '.join(component.get('modifiers', []))}
            Return Type: {component.get('return_type', 'Not specified')}
            Parameters: {', '.join(f"{p.get('name')}: {p.get('type')}" for p in component.get('parameters', []))}
            Throws: {', '.join(component.get('throws', []))}
            Enclosing Class: {', '.join(component.get('enclosing_class', []))}
            """

        elif component_type == "interfaces":
            details["name"]=component.get('interface_name', 'Not specified')
            details["component_details"] = f"""
            Interface Name: {component.get('interface_name', 'Not specified')}
            Modifiers: {', '.join(component.get('modifiers', []))}
            Methods: {', '.join(component.get('methods', []))}
            Default Methods: {', '.join(component.get('default_methods', []))}
            Enclosing Class: {', '.join(component.get('enclosing_class', []))}
            """

        elif component_type == "constructors":
            details["name"]=component.get('constructor_name', 'Not specified')
            details["component_details"] = f"""
            Constructor Name: {component.get('constructor_name', 'Not specified')}
            Modifiers: {', '.join(component.get('modifiers', []))}
            Parameters: {', '.join(f"{p.get('name')}: {p.get('type')}" for p in component.get('parameters', []))}
            Throws: {', '.join(component.get('throws', []))}
            Enclosing Class: {', '.join(component.get('enclosing_class', []))}
            """

        elif component_type == "fields":
            details["name"]=component.get('field_name', 'Not specified')
            details["component_details"] = f"""
            Field Name: {component.get('field_name', 'Not specified')}
            Type: {component.get('type', 'Not specified')}
            Modifiers: {', '.join(component.get('modifiers', []))}
            Initial Value: {component.get('initial_value', 'Not specified')}
            Enclosing Class: {', '.join(component.get('enclosing_class', []))}
            """
        details['dependencies']=component.get('dependencies', 'Not provided')
        details["implementation"] = component.get('se', 'Not provided')
        details["description"] = component.get('description', 'Not provided')
        details["first_Pass_Analysis"] = component.get('first_Pass_Analysis', 'Not provided')
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
                extracted_components = ast.literal_eval(json_string)

            # Initialize the analyzed components structure
            analyzed_components = {
                "classes": [],
                "interfaces": [],
                "methods": [],
                "constructors": [],
                "fields": []
            }

            # Step 3: Analyze components
            print("Analyzing components...")
            for comp_type in analyzed_components.keys():
                for comp in extracted_components.get(comp_type, []):
                    # First pass analysis
                    analyzed_comp = self.first_pass_analyze_component(comp, comp_type)
                    # Second pass analysis on the same component
                    analyzed_comp = self.second_pass_analyze_component(analyzed_comp, comp_type)
                    # Add the fully analyzed component
                    analyzed_components[comp_type].append(analyzed_comp)

            return analyzed_components

        except Exception as e:
            print(f"Error in process_and_analyze: {str(e)}")
            return None

if __name__ == "__main__":
    extractor = JavaCodeExtractor(
        model_name="llama-3.3-70b-versatile",
        api_key="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"
    )
    java_file_path = 'code.java'
    
    # Process and analyze (this creates the PKL file)
    analyzed_components = extractor.process_and_analyze(java_file_path)
    
    if analyzed_components:
        # Clean the result for output
        cleaned_result = extractor.remove_implementation(analyzed_components)
        
        # Save to JSON file
        with open('output.json', 'w') as f:
            json.dump(cleaned_result, f, indent=2)
        
        # Print the result
        print(json.dumps(cleaned_result, indent=2))
    else:
        print("Analysis failed to complete successfully.")
