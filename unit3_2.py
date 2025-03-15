from langchain_groq import ChatGroq
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import ast
import re
import traceback
import datetime
from unit1 import JavaSyntaxFixer
from unit2_1 import JavaCodeExtractor as JsonExtractor
import concurrent.futures
import multiprocessing
from typing import List, Dict, Any
#! Alert : needs to be exmined 
#^ TODO: changes to be made 
#& RAG RELATED COMMENTS 
#~ SUMMARY: SUMMRY OF EACH FUNCTION
#? UNSURE: SEEMS CORRECT MOSTLY
#* General Comments 


class DebugLogger:
    """Helper class for logging debug information to a file."""
    def __init__(self, debug_file="debug_log.txt"):
        self.debug_file = debug_file
        #*Initialize log file with timestamp
        with open(self.debug_file, 'w') as f:
            f.write(f"=== Debug Log Started at {datetime.datetime.now()} ===\n\n")
    
    def log(self, message, data=None):
        """Log a message with optional data to the debug file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.debug_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
            if data is not None:
                if isinstance(data, dict) or isinstance(data, list):
                    f.write(json.dumps(data, indent=2, default=str) + "\n")
                else:
                    f.write(str(data) + "\n")
            f.write("\n" + "-"*80 + "\n\n")

class JavaCodeAnalyzer:
    
    #~ INITIALIZATION FUNCTION(NO AMBIGUITY, FULLY FUNCTIONAL, 0 CHANGES REQUIRED)
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.2):
        self.logger = DebugLogger(debug_file="java_analyzer_debug.log")
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
        
        #*Store all components for cross-referencing
        self.all_components = {}
        
        #*Initialize debug logger
        
    #~ FUNCTION TO LOAD MEMORY(REQUIRES FURTHER INVESTIGATION. IT SEEMS TO BE WORKING FINE HOWEVER I HAVE NOT EXAMINED THIS PORTION OF THE CODE. WILL REQUIRE MOST DEBUGGING FOR FURTHER ANALYSIS)
    def _load_memory(self):
        """Load or initialize the memory dictionary with all required keys."""
        default_memory = {
            'embeddings': [], 
            'texts': [], 
            'component_types': [],
            'component_names': [],
            'analysis_types': []  #*"first_pass", "second_pass", "evaluation"
        }
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    loaded_memory = pickle.load(f)
                    
                self.logger.log("Memory file loaded", {
                    "keys": list(loaded_memory.keys()),
                    "embeddings_count": len(loaded_memory.get('embeddings', [])),
                    "texts_count": len(loaded_memory.get('texts', []))
                })
                    
                #*Ensure all required keys exist in the loaded memory
                for key in default_memory:
                    if key not in loaded_memory:
                        loaded_memory[key] = []
                        self.logger.log(f"Added missing memory key: {key}")
                        
                #*If the loaded memory has different lengths of arrays, reset it
                lengths = [len(loaded_memory[key]) for key in default_memory]
                if len(set(lengths)) > 1:  #*If lengths aren't all the same
                    self.logger.log("Memory file has inconsistent data lengths. Resetting memory.", {
                        "lengths": {key: len(loaded_memory[key]) for key in default_memory}
                    })
                    return default_memory
                    
                return loaded_memory
            except Exception as e:
                self.logger.log(f"Error loading memory file: {str(e)}. Creating new memory.")
                return default_memory
                
        self.logger.log("Memory file not found. Creating new memory.")
        return default_memory
    
    #~ FUNCTION TO LOAD MEMORY INTO A PKL FILE(NO AMBIGUITY, FULLY FUNCTIONAL, 0 CHANGES REQUIRED)
    def _save_memory(self):
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)
    
    #~ FUNCTION TO UPDATE MEMORY(REQUIRES FURTHER INVESTIGATION. IT SEEMS TO BE WORKING FINE HOWEVER I HAVE NOT EXAMINED THIS PORTION OF THE CODE. WILL REQUIRE MOST DEBUGGING FOR FURTHER ANALYSIS)
    def update_memory(self, component_type: str, component: dict, analysis: str, analysis_type: str = "second_pass"):
        """Store component analysis in vector memory with improved metadata."""
        try:
            #*Get component name based on type
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
            
            self.logger.log(f"Updating memory for {component_type}: {component_name}", {
                "analysis_type": analysis_type,
                "analysis_length": len(analysis)
            })
            
            #*Create a rich context string
            context = f"""
            Type: {component_type}
            Name: {component_name}
            Analysis: {analysis}
            Description: {component.get('description', '')}
            """
            
            #*Generate embedding
            self.logger.log("Generating embedding for context")
            embedding = self.embedding_model.encode(context)
            
            #*Store in memory with enhanced metadata
            self.memory['embeddings'].append(embedding.tolist())
            self.memory['texts'].append(context)
            self.memory['component_types'].append(component_type)
            self.memory['component_names'].append(component_name)
            self.memory['analysis_types'].append(analysis_type)
            
            #*Log memory statistics
            self.logger.log("Memory updated", {
                "total_entries": len(self.memory['embeddings']),
                "embeddings_dimension": len(embedding) if isinstance(embedding, np.ndarray) else "unknown"
            })
            
        #     #*Save to file
        #     self._save_memory()
        #     self.logger.log("Memory saved to file")
        # except Exception as e:
        #     self.logger.log(f"Warning: Failed to update memory: {str(e)}")
        #     self.logger.log(traceback.format_exc())

        #* Don't save memory after every update - only save periodically to improve performance
        #* We'll save every 10 updates or when analysis_type is "evaluation" (end of cycle)
            self._memory_updates_since_save = getattr(self, '_memory_updates_since_save', 0) + 1
        
            if analysis_type == "evaluation" or self._memory_updates_since_save >= 10:
                self._save_memory()
                self._memory_updates_since_save = 0
                self.logger.log("Memory saved to file (batch operation)")
        
        except Exception as e:
            self.logger.log(f"Warning: Failed to update memory: {str(e)}")
            self.logger.log(traceback.format_exc())
        
    #~ FUNCTION TO CALCULATE COSINE SIMILARITY(NO AMBIGUITY, FULLY FUNCTIONAL, 0 CHANGES REQUIRED)
    def cosine_similarity(self, a, b):
        try:
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except Exception as e:
            self.logger.log(f"Warning: Similarity calculation failed: {str(e)}")
            return 0.0
    
    #~ FUNCTION TO SEARCH FOR SEMANTIC ANALYSIS OF RELATED COMPONENTS(REQUIRES FURTHER INVESTIGATION. IT SEEMS TO BE WORKING FINE HOWEVER I HAVE NOT EXAMINED THIS PORTION OF THE CODE. WILL REQUIRE MOST DEBUGGING FOR FURTHER ANALYSIS)
    def search_similar_analyses(self, query: str, component_type: str = None, top_k: int = 3, analysis_type: str = None):
        """
        Search for similar previous analyses with filtering by component type and analysis type.
        """
        query_embedding = self.embedding_model.encode(query)
        similarities = []
        
        #*Ensure memory has all required keys
        required_keys = ['embeddings', 'texts', 'component_types', 'component_names', 'analysis_types']
        for key in required_keys:
            if key not in self.memory:
                self.memory[key] = []
                
        #*Return empty list if no embeddings
        if not self.memory['embeddings']:
            return []
        
        #*Check if all arrays have the same length
        lengths = [len(self.memory[key]) for key in required_keys]
        if len(set(lengths)) > 1:
            self.logger.log("Warning: Memory data has inconsistent lengths. Returning empty results.", {
                "lengths": {key: len(self.memory[key]) for key in required_keys}
            })
            return []
            
        for idx, stored_embedding in enumerate(self.memory['embeddings']):
            #*Skip if index is out of bounds for any array
            if (idx >= len(self.memory['component_types']) or 
                idx >= len(self.memory['analysis_types']) or
                idx >= len(self.memory['texts'])):
                continue
                
            #*Apply filters if specified
            if component_type and self.memory['component_types'][idx] != component_type:
                continue
                
            if analysis_type and self.memory['analysis_types'][idx] != analysis_type:
                continue
                
            score = self.cosine_similarity(query_embedding, stored_embedding)
            
            #*Safe access to component name
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
    
    #~ RETURNS THE DETAILS(MOST RELEVANT METADATA AND DEPENDENCIES FOR EACH COMPONENT)(NO AMBIGUITY, FULLY FUNCTIONAL,  1 CHANGE MAY BE REQUIRED)
    def get_related_components(self, component: dict, component_type: str):
        """
        Find related components based on dependencies.
        Returns a dict of related components grouped by relationship type.
        """
        related = {}
        
        #*Skip if no dependencies
        
        if 'dependencies' not in component or not component['dependencies']:
            return related
            
        dependencies = component['dependencies']
        #? STORES ONLY A SUBSET OF RELATED COMPONENT DETAILS FROM METADATA AND DEPENDENCIES(MAY NOT BE THE MOST RELEVANT)
        if component_type == "classes":
            #*Find super classes
            if 'super_class' in component and component['super_class']:
                related['super_classes'] = self._find_components('classes', component['super_class'])
                
            #*Find implemented interfaces
            if 'implemented_interfaces' in component and component['implemented_interfaces']:
                related['interfaces'] = self._find_components('interfaces', component['implemented_interfaces'])
                
            #*Find class methods
            if 'methods' in component and component['methods']:
                related['methods'] = self._find_components('methods', component['methods'])
                
            #*Find class fields
            if 'fields' in component and component['fields']:
                related['fields'] = self._find_components('fields', component['fields'])
                
            #*Find nested classes
            if 'nested_classes_or_interfaces' in component and component['nested_classes_or_interfaces']:
                related['nested_classes'] = self._find_components('classes', component['nested_classes_or_interfaces'])
                
            #*Find constructors
            if 'constructors' in component and component['constructors']:
                related['constructors'] = self._find_components('constructors', component['constructors'])
        #? STORES ONLY A SUBSET OF RELATED COMPONENT DETAILS FROM METADATA AND DEPENDENCIES(MAY NOT BE THE MOST RELEVANT)
        elif component_type == "methods":
            #*Find called methods
            if 'called_methods' in dependencies and dependencies['called_methods']:
                related['called_methods'] = self._find_components('methods', dependencies['called_methods'])
                
            #*Find used fields
            if 'used_fields' in dependencies and dependencies['used_fields']:
                related['used_fields'] = self._find_components('fields', dependencies['used_fields'])
                
            #*Find method's class
            if 'enclosing_class' in component and component['enclosing_class']:
                related['enclosing_class'] = self._find_components('classes', component['enclosing_class'])
                
            #*Find overridden methods
            if 'overrides' in dependencies and dependencies['overrides']:
                related['overridden_methods'] = self._find_components('methods', [dependencies['overrides']])
                
            #*Find implemented interface methods
            if 'implements' in dependencies and dependencies['implements']:
                related['implemented_methods'] = self._find_components('methods', [dependencies['implements']])
        #? STORES ONLY A SUBSET OF RELATED COMPONENT DETAILS FROM METADATA AND DEPENDENCIES(MAY NOT BE THE MOST RELEVANT)
        elif component_type == "interfaces":
            #*Find methods defined in interface
            if 'methods' in component and component['methods']:
                related['interface_methods'] = self._find_components('methods', component['methods'])
                
            #*Find default methods
            if 'default_methods' in component and component['default_methods']:
                related['default_methods'] = self._find_components('methods', component['default_methods'])
                
            #*Find enclosing class if nested interface
            if 'enclosing_class' in component and component['enclosing_class']:
                related['enclosing_class'] = self._find_components('classes', component['enclosing_class'])
                
            #*Find implementing classes
            if 'implementing_classes' in dependencies and dependencies['implementing_classes']:
                related['implementing_classes'] = self._find_components('classes', dependencies['implementing_classes'])
                
            #*Find extended interfaces
            if 'extended_interfaces' in dependencies and dependencies['extended_interfaces']:
                related['extended_interfaces'] = self._find_components('interfaces', dependencies['extended_interfaces'])
        #? STORES ONLY A SUBSET OF RELATED COMPONENT DETAILS FROM METADATA AND DEPENDENCIES(MAY NOT BE THE MOST RELEVANT) 
        elif component_type == "constructors":
            #*Find enclosing class
            if 'enclosing_class' in component and component['enclosing_class']:
                related['enclosing_class'] = self._find_components('classes', component['enclosing_class'])
                
            #*Find chained constructors
            if 'constructor_chaining' in dependencies and dependencies['constructor_chaining']:
                related['chained_constructors'] = self._find_components('constructors', [dependencies['constructor_chaining']])
                
            #*Find called methods
            if 'called_methods' in dependencies and dependencies['called_methods']:
                related['called_methods'] = self._find_components('methods', dependencies['called_methods'])
                
            #*Find initialized fields
            if 'initialized_fields' in dependencies and dependencies['initialized_fields']:
                related['initialized_fields'] = self._find_components('fields', dependencies['initialized_fields'])
        #? STORES ONLY A SUBSET OF RELATED COMPONENT DETAILS FROM METADATA AND DEPENDENCIES(MAY NOT BE THE MOST RELEVANT)   
        elif component_type == "fields":
            #*Find enclosing class
            if 'enclosing_class' in component and component['enclosing_class']:
                related['enclosing_class'] = self._find_components('classes', component['enclosing_class'])
                
            #*Find methods using this field
            if 'used_by_methods' in dependencies and dependencies['used_by_methods']:
                related['using_methods'] = self._find_components('methods', dependencies['used_by_methods'])
                
            #*Find shadowed fields
            if 'shadows_field' in dependencies and dependencies['shadows_field']:
                related['shadowed_fields'] = self._find_components('fields', [dependencies['shadows_field']])
                
            #*Find initialization dependencies
            if 'initialization_dependencies' in dependencies and dependencies['initialization_dependencies']:
                #*These could be either fields or methods
                field_deps = self._find_components('fields', dependencies['initialization_dependencies'])
                method_deps = self._find_components('methods', dependencies['initialization_dependencies'])
                
                if field_deps:
                    related['dependent_fields'] = field_deps
                    
                if method_deps:
                    related['dependent_methods'] = method_deps
        comp_name=self._get_component_name(component, component_type)
        self.logger.log(f"related for {component_type}: {comp_name}", related)         
        return related
   
    #~ FUNCTION TO RETURN THE METADATA OF THE RELATD COMPONENTS (NOT THE DEPENDENCIES AS THI IS BEING STORED IN THE ABOVE FUNCTION)(NO AMBIGUITY, FULLY FUNCTIONAL, 0 CHANGES REQUIRED) 
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
        
        comp_name=self._get_component_name(component, component_type)
        self.logger.log(f"result for {component_type}: {comp_name}", result)   
        return result
    
    #~ FUNCTION TO PERFORM FIRST PASS ANALYSIS OF A COMPONENT(NO AMBIGUITY, FULLY FUNCTIONAL, 2 CHANGES REQUIRED, 1 CHANGE MAY BE REQUIRED)
    def first_pass_analyze_component(self, component: dict, component_type: str):
        """
        First pass analysis focusing on individual component properties.
        """
        component_name = self._get_component_name(component, component_type)
        self.logger.log(f"Starting first pass analysis for {component_type}: {component_name}")
        
        details = self.prepare_analysis_prompt(component, component_type)
        self.logger.log("Analysis prompt details prepared", {
            "name": details['name'],
            "has_implementation": bool(details.get('implementation')),
            "description_length": len(details.get('description', ''))
        })
        
        # *Search for similar analyses from first pass only (with safe implementation)
        # ^ QUERY FORMAT CAN BE IMPROVED
        # query = f"{component_type} {details['name']} {details['description']}"
        # self.logger.log(f"Searching for similar analyses", {"query": query})
        
        # similar_analyses = self.search_similar_analyses(
        #     query, 
        #     component_type=component_type, 
        #     analysis_type="first_pass"
        # )
        
        # self.logger.log("Similar analyses search results", {
        #     "found_count": len(similar_analyses)
        # })
        
        # #*Safely handle potentially empty or differently structured similar_analyses
        # previous_context = ""
        # #? seems correct (not sure as rag functions are yet to be examined)
        # if similar_analyses:
        #     try:
        #         previous_context = "\n".join([text for _, text, _, _ in similar_analyses])
        #         self.logger.log("Previous context extracted", {"length": len(previous_context)})
        #     except (ValueError, TypeError) as e:
        #         #*Fallback for different structure
        #         self.logger.log(f"Error extracting previous context: {str(e)}")
        #         previous_context = "\n".join([str(item) for item in similar_analyses])
        # #^ PROMPT INSTRUCTIONS CAN BE IMPROVED. eg any potential issues in the semantics of the code strictly and not the ones regardig readability and structure
        prompt = f"""
        You are evaluating a Java {component_type} for a lab test submission.
        Perform a HIGH-LEVEL ANALYSIS of the following component:

        Component Name: {details['name']}
        
        Component Details:
        {details['component_details']}
        
        Implementation:
        {details['implementation']}
        
        TASK: Provide a brief high-level analysis using EXACTLY the following format:

        PURPOSE:
        [1-2 sentences describing the main functionality of the component]

        ALGORITHM:
        [Name and brief description of the algorithm/approach used]

        POTENTIAL ISSUES:
        - [Bullet point listing a potential logical error or concern, if any]
        - [Another issue, if applicable]
        
        COMPLETENESS:
        [1 sentence stating whether the component appears to implement required functionality]

        Keep your entire response under 100 words total. Focus only on observable functional aspects.
        DO NOT comment on coding style, formatting, or documentation unless it impacts logical correctness.
        """
        
        self.logger.log("Sending prompt to ChatGroq", {"prompt_length": len(prompt)})
        response = self.chatgroq.invoke(prompt)
        try:
            analysis = response.content.strip()
            self.logger.log("Received analysis response", {"analysis_length": len(analysis)})
        except Exception as e:
            self.logger.log(f"Error getting analysis content: {str(e)}")
            analysis = "Analysis could not be generated."
        
        component["first_pass_analysis"] = analysis
        self.update_memory(component_type, component, analysis, "first_pass")
        
        self.logger.log(f"Completed first pass analysis for {component_type}: {component_name}")
        return component
    
    #~ FUNCTION TO PERFORM SECOND PASS ANALYSIS OF A COMPONENT(NO AMBIGUITY, FULLY FUNCTIONAL, 2 CHANGES REQUIRED, 1 CHANGE MAY BE REQUIRED)
    def second_pass_analyze_component(self, component: dict, component_type: str):
        """
        Second pass analysis focusing on component relationships and cross-references.
        """
        component_name = self._get_component_name(component, component_type)
        self.logger.log(f"Starting second pass analysis for {component_type}: {component_name}")
        
        details = self.prepare_analysis_prompt(component, component_type)
        
        #*Get related components
        self.logger.log("Finding related components")
        related_components = self.get_related_components(component, component_type)
        self.logger.log("Related components found", {
            "relation_types": list(related_components.keys()),
            "total_related": sum(len(comps) for comps in related_components.values())
        })
        
        #*Build related components context
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
                    self.logger.log(f"Error processing related component: {str(e)}")
                    related_context += f"- Error retrieving component: {str(e)}\n"
        
        #*Search for similar analyses from second pass only
        #^ QUERY FORMAT CAN BE IMPROVED
        query = f"{component_type} {details['name']} {details['description']}"
        self.logger.log("Searching for similar second pass analyses", {"query": query})
        #? seems correct (not sure as rag functions are yet to be examined)
        similar_analyses = self.search_similar_analyses(
            query, 
            component_type=component_type, 
            analysis_type="second_pass"
        )
        self.logger.log("Found similar analyses", {"count": len(similar_analyses)})
        
        #*Safely handle potentially empty or differently structured similar_analyses
        previous_context = ""
        if similar_analyses:
            try:
                previous_context = "\n".join([text for _, text, _, _ in similar_analyses])
                self.logger.log("Previous context extracted", {"length": len(previous_context)})
            except (ValueError, TypeError) as e:
                self.logger.log(f"Error extracting previous context: {str(e)}")
                #*Fallback for different structure
                previous_context = "\n".join([str(item) for item in similar_analyses])
        self.logger.log("Related context and previous context extracted" , {"related_context": related_context, "previous_context": previous_context})
        
        #^ PROMPT INSTRUCTIONS CAN BE IMPROVED. eg any potential issues in the semantics of the code strictly and not the ones regardig readability and structure
        prompt = f"""
        You are evaluating a Java {component_type} for a lab test submission. 
        Perform an IN-DEPTH ALGORITHM ANALYSIS:

        COMPONENT: {details['name']} ({component_type})

        Initial Analysis Results:
        {component.get('first_pass_analysis', 'Not available')}
        
        Component Details:
        {details['component_details']}
        
        Dependencies:
        {self._format_dependencies(component.get('dependencies', {}))}
        
        Related Components:
        {related_context if related_context else "No related components found."}
        
        Implementation:
        {details['implementation']}
        
        TASK: Analyze this implementation using EXACTLY the following format:

        LOGICAL CORRECTNESS:
        [2-3 sentences about whether the algorithm is logically correct]
        - Strengths: [List 1-2 key strengths in the logical implementation]
        - Weaknesses: [List 1-2 key weaknesses or edge cases that aren't handled]
        
        TIME COMPLEXITY:
        - Overall: O(X) where X is [explanation]
        - Bottleneck: [Identify the specific operation that determines the time complexity]
        - Potential Improvement: [If applicable, briefly describe how time complexity could be improved]
        
        SPACE COMPLEXITY:
        - Overall: O(X) where X is [explanation]
        - Key Factors: [What specific data structures or operations contribute to this complexity]
        
        ALGORITHM CHOICE:
        [1-2 sentences on whether this is an appropriate algorithm for the task]
        [If applicable, briefly describe a more efficient alternative]

        Always provide specific line numbers or code snippets to support your analysis.
        Focus exclusively on logical correctness and computational efficiency.
        """
        
        self.logger.log("Sending prompt to ChatGroq", {"prompt_length": len(prompt)})
        response = self.chatgroq.invoke(prompt)
        try:
            analysis = response.content.strip()
            self.logger.log("Received analysis response", {"analysis_length": len(analysis)})
        except Exception as e:
            self.logger.log(f"Error getting analysis content: {str(e)}")
            analysis = "Analysis could not be generated."
        
        component["second_pass_analysis"] = analysis
        self.update_memory(component_type, component, analysis, "second_pass")
        
        self.logger.log(f"Completed second pass analysis for {component_type}: {component_name}")
        return component
    
    #~ FUNCTION TO SCORE AND EVALUATE A COMPONENT(NO AMBIGUITY, FULLY FUNCTIONAL, 2 ALERT CHANGES REQUIRED)
    def evaluate_component(self, component: dict, component_type: str):
        """
        Final evaluation focusing on scoring, best practices, and preparing for comparison.
        """
        component_name = self._get_component_name(component, component_type)
        self.logger.log(f"Starting evaluation for {component_type}: {component_name}")
        
        details = self.prepare_analysis_prompt(component, component_type)
        
        first_pass = component.get('first_pass_analysis', 'Not available')
        second_pass = component.get('second_pass_analysis', 'Not available')
        
        # #*Safely retrieve similar analyses for evaluation
        # query = f"{component_type} {details['name']} evaluation"
        # self.logger.log("Searching for similar evaluations", {"query": query})
        # similar_analyses = self.search_similar_analyses(
        #     query, 
        #     component_type=component_type, 
        #     analysis_type="evaluation"
        # )
        # self.logger.log("Found similar evaluations", {"count": len(similar_analyses)})
        
        # #*Safely extract previous context
        # #! PREVIOUS CONTEXT IS NOT REQUIRED FOR EVALUATION. TO BE REMOVED IMMEDIATELY AFTER DISCUSSION. NOT REMOVING COZ IT MAY BE REQUIRED FOR FUTURE USE EXAMPLE TO MAINTAIN CONISTENCY ACROSS SCORING AND EVALUATION
        # previous_context = ""
        
        # if similar_analyses:
        #     try:
        #         previous_context = "\n".join([text for _, text, _, _ in similar_analyses])
        #         self.logger.log("Previous evaluation context extracted", {"length": len(previous_context)})
        #     except (ValueError, TypeError) as e:
        #         self.logger.log(f"Error extracting previous context: {str(e)}")
        #         #*Fallback for different structure
        #         previous_context = "\n".join([str(item) for item in similar_analyses])

        #! PROMPT NEEDS TO BE CHANGED AFTER DICUSSION . A RUBRIC CAN GO HERE. 
        prompt = f"""
        You are evaluating a Java {component_type} for a lab test submission. 
        Provide a FINAL EVALUATION using the following rubric and format:

        COMPONENT: {details['name']} ({component_type})

        First Pass Analysis:
        {first_pass}
        
        Second Pass Analysis:
        {second_pass}
        
        Implementation:
        {details['implementation']}
        
        EVALUATION:
        
        LOGICAL CORRECTNESS: [SCORE]/5
        - Justification: [1-2 sentences with specific examples]
        - To improve: [1 specific recommendation]
        
        TIME COMPLEXITY: [SCORE]/3
        - Justification: [1 sentence with Big-O notation]
        - To improve: [1 specific recommendation if score < 3]
        
        SPACE COMPLEXITY: [SCORE]/2
        - Justification: [1 sentence with Big-O notation]
        - To improve: [1 specific recommendation if score < 2]
        
        CODE QUALITY: [SCORE]/2
        - Justification: [1 sentence focusing only on structural issues that affect algorithm correctness]
        - To improve: [1 specific recommendation]
        
        TOTAL SCORE: [SUM]/12
        
        KEY STRENGTHS:
        - [Key strength 1]
        - [Key strength 2 if applicable]
        
        KEY RECOMMENDATIONS:
        - [Most important recommendation]
        - [Second most important recommendation]

        Use the scoring guidelines from the rubric:
        - LOGICAL CORRECTNESS (0-5): 5=perfect, 4=minor issues, 3=works with limitations, 2=significant issues, 1=major flaws, 0=incorrect
        - TIME COMPLEXITY (0-3): 3=optimal, 2=reasonable, 1=suboptimal, 0=poor
        - SPACE COMPLEXITY (0-2): 2=optimal, 1=reasonable, 0=poor
        - CODE QUALITY (0-2): 2=excellent structure, 1=adequate structure, 0=poor structure affecting correctness

        Be precise and technical. Focus on algorithmic aspects rather than style.
        """

        
        self.logger.log("Sending evaluation prompt to ChatGroq", {"prompt_length": len(prompt)})
        response = self.chatgroq.invoke(prompt)
        try:
            evaluation = response.content.strip()
            self.logger.log("Received evaluation response", {"evaluation_length": len(evaluation)})
        except Exception as e:
            self.logger.log(f"Error getting evaluation content: {str(e)}")
            evaluation = "Evaluation could not be generated."
        
        component["evaluation"] = evaluation
        self.update_memory(component_type, component, evaluation, "evaluation")
        
        #*Try to extract and log the score for monitoring
        try:
            score_match = re.search(r'COMPONENT SCORE.*?(\d+(?:\.\d+)?)', evaluation, re.DOTALL)
            if score_match:
                score = float(score_match.group(1))
                self.logger.log(f"Component score extracted", {"score": score})
        except Exception as e:
            self.logger.log(f"Could not extract component score: {str(e)}")
        
        self.logger.log(f"Completed evaluation for {component_type}: {component_name}")
        return component
    
    #~ FUNCTION TO GET COMPONENT TYPE(NO AMBIGUITY, FULLY FUNCTIONAL, O CHANGES REQUIRED)  
    def _get_component_type(self, component: dict) -> str:
        """Determine the type of a component based on its keys."""
        if not isinstance(component, dict):
            return 'unknown'
            
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
        
    #~ FUNCTION TO GET COMPONENT NAME(NO AMBIGUITY, FULLY FUNCTIONAL, O CHANGES REQUIRED)     
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
    
    #~ FUNCTION TO APPEND AND FORMAT DEPENDENCIES(NEEDS TO BE EXAMINED, HOWEVER IT SEEMS TO BE WORKING FINE)
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
    
    #~ FUNCTION TO GET COMPONENT DETAILS(NO AMBIGUITY, FULLY FUNCTIONAL, 1 CHANGE REQUIRED)
    def prepare_analysis_prompt(self, component: dict, component_type: str):
        """Prepares the structured analysis details for a Java component."""
        details = {}
        
        #*Helper function to safely join values
        def safe_join(values):
            if isinstance(values, list):
                return ', '.join(values)
            elif isinstance(values, str):
                return values
            elif values is None:
                return 'None'
            else:
                return str(values)
        
        #*Helper to safely get parameters string
        def get_params_str(params):
            if not params or not isinstance(params, list):
                return ''
            return ', '.join(f"{p.get('name', '')}: {p.get('type', '')}" for p in params)

        if component_type == "classes":
            details["name"] = component.get('class_name', 'Not specified')
            #^ I think for other details specific to the component types we should probably store it in a list like the common details such as implementation, description, dependencies
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
        self.logger.log(f"Analysis prompt prepared for {component_type}: {details['name']}", details)
        return details
    
    #~REMOVES IMPLEMENTATION OF EACH COMPONENT FROM THE JSON AS IT IS NOT REQUIRED(NO AMBIGUITY, FULLY FUNCTIONAL, 0 CHANGES REQUIRED)
    def remove_implementation(self, components_dict):
        """Removes the implementation field from all components in the dictionary."""
        for component_type in components_dict:
            if isinstance(components_dict[component_type], list):
                for component in components_dict[component_type]:
                    if isinstance(component, dict) and 'implementation' in component:
                        component.pop('implementation')
        return components_dict
    
        #! I DO NOT THINK THIS FUNCTION IS REQUIRED. KEEPING IT FOR FUTURE USE. APART FORM THAT IT IS FUNCTIONAL AND IT GENERATES A SUMMARY OF THE ENTIRE EVALUATIONINCLUDING FINAL SCORE
    def generate_summary_report(self, analyzed_components):
        """
        Generate a summary report of the analysis.
        """
        self.logger.log("Generating summary report")
        
        #*Collect component counts
        component_counts = {comp_type: len(comps) for comp_type, comps in analyzed_components.items() if comps}
        self.logger.log("Component counts", component_counts)
        
        #*Collect all individual component scores
        all_scores = []
        score_by_type = {}
        algorithm_issues = []
        complexity_issues = []
        
        for comp_type, comps in analyzed_components.items():
            type_scores = []
            for comp in comps:
                if 'evaluation' in comp:
                    #*Try to extract score from evaluation
                    eval_text = comp['evaluation']
                    score_match = re.search(r'TOTAL SCORE.*?(\d+(?:\.\d+)?)', eval_text, re.DOTALL)
                    if score_match:
                        try:
                            score = float(score_match.group(1))
                            all_scores.append(score)
                            type_scores.append(score)
                            
                            #*Log the component name and its score
                            comp_name = self._get_component_name(comp, comp_type)
                            self.logger.log(f"Score for {comp_type} {comp_name}: {score}")
                            
                            #*Collect algorithm correctness issues - with safer regex
                            if "LOGICAL CORRECTNESS" in eval_text:
                                correctness_section = re.search(r'LOGICAL CORRECTNESS.*?(?=TIME COMPLEXITY|$)', eval_text, re.DOTALL)
                                if correctness_section:
                                    # Try to find a score pattern, but don't fail if not found
                                    score_pattern = re.search(r'(\d+)(?:/|\s*out\s*of\s*)5', correctness_section.group(0))
                                    if score_pattern and int(score_pattern.group(1)) < 4:
                                        algorithm_issues.append(f"{comp_name}: {correctness_section.group(0).strip()}")
                                    # If we can't find the score pattern, still include any section with "error", "incorrect", or "issue"
                                    elif re.search(r'error|incorrect|issue|fail|bug', correctness_section.group(0), re.IGNORECASE):
                                        algorithm_issues.append(f"{comp_name}: {correctness_section.group(0).strip()}")
                            
                            #*Collect complexity issues - with safer regex
                            if "TIME COMPLEXITY" in eval_text or "SPACE COMPLEXITY" in eval_text:
                                complexity_section = re.search(r'(TIME COMPLEXITY|SPACE COMPLEXITY).*?(SPACE COMPLEXITY|CODE QUALITY|$)', eval_text, re.DOTALL)
                                if complexity_section:
                                    # Try to find a score pattern, but don't fail if not found
                                    time_score = re.search(r'(\d+)(?:/|\s*out\s*of\s*)3', complexity_section.group(0))
                                    space_score = re.search(r'(\d+)(?:/|\s*out\s*of\s*)2', complexity_section.group(0))
                                    
                                    # Include if score is low or contains specific keywords
                                    if ((time_score and int(time_score.group(1)) < 2) or 
                                        (space_score and int(space_score.group(1)) < 1) or
                                        re.search(r'inefficient|slow|suboptimal|improve|bottleneck', complexity_section.group(0), re.IGNORECASE)):
                                        complexity_issues.append(f"{comp_name}: {complexity_section.group(0).strip()}")
                        except (ValueError, AttributeError) as e:
                            self.logger.log(f"Error processing score for {comp_type}: {str(e)}")
            
            #*Calculate average score for this component type
            if type_scores:
                avg_type_score = sum(type_scores) / len(type_scores)
                score_by_type[comp_type] = avg_type_score
                self.logger.log(f"Average score for {comp_type}: {avg_type_score}")
        
        #*Calculate average score if available
        avg_score = sum(all_scores) / len(all_scores) if all_scores else None
        if avg_score:
            self.logger.log(f"Overall average score: {avg_score}")
        else:
            self.logger.log("No scores available")
        
        #*Create list of all components
        components_list = []
        for comp_type, comps in analyzed_components.items():
            if comps:
                for comp in comps:
                    name = self._get_component_name(comp, comp_type)
                    components_list.append(f"{name} ({comp_type})")
        
        components_text = "\n".join(f"- {comp}" for comp in components_list)
        self.logger.log("Generated components list", {"count": len(components_list)})
        
        #*Create summary of noted issues
        algorithm_issues_text = "\n".join(algorithm_issues[:5]) if algorithm_issues else "No major correctness issues identified." #*Limit to top 5 issues
        complexity_issues_text = "\n".join(complexity_issues[:5]) if complexity_issues else "No major complexity issues identified." #*Limit to top 5 issues
        
        #! IMPROVED PROMPT AFTER DISCUSSION - FOCUSES ON ALGORITHM CORRECTNESS, TIME/SPACE COMPLEXITY & CODE QUALITY
        prompt = f"""
        Generate a comprehensive summary report for a Java lab test submission focused on algorithmic correctness and efficiency.
        
        Components analyzed:
        {components_text}
        
        Component counts: {json.dumps(component_counts)}
        
        Overall average score: {avg_score if avg_score is not None else 'Not available'}/12
        
        Scores by component type: {json.dumps(score_by_type) if score_by_type else 'Not available'}
        
        Key algorithm correctness issues identified:
        {algorithm_issues_text}
        
        Key complexity issues identified:
        {complexity_issues_text}
        
        Use EXACTLY the following format for your report:

        # LAB TEST EVALUATION SUMMARY
        
        ## ALGORITHM ASSESSMENT (40%)
        
        ### Correctness Analysis:
        [2-3 sentences evaluating overall algorithm correctness]
        
        ### Critical Issues:
        - [Specific issue #1 with affected component]
        - [Specific issue #2 with affected component]
        
        ### Correctness Score: [X]/40
        
        ## COMPUTATIONAL EFFICIENCY (40%)
        
        ### Time Complexity:
        - Overall: [Big-O notation for key components]
        - Key bottlenecks: [List the most significant performance bottlenecks]
        
        ### Space Complexity:
        - Overall: [Big-O notation for key components]
        - Main concerns: [List any memory usage issues]
        
        ### Efficiency Score: [X]/40
        
        ## CODE STRUCTURE (20%)
        
        ### Structural Impact on Algorithms:
        [2-3 sentences on how code structure affects algorithm implementation]
        
        ### Structure Score: [X]/20
        
        ## FINAL ASSESSMENT
        
        ### Total Score: [X]/100
        
        ### Top 3 Recommendations:
        1. [Most important recommendation]
        2. [Second recommendation]
        3. [Third recommendation]
        
        ### Path to Perfect Score:
        [1-2 sentences explaining what would be required for a perfect score]
        
        Keep your assessment technical and focused on algorithmic aspects rather than stylistic concerns.
        Use precise terminology from algorithms and data structures in your evaluation.
        """
        
        self.logger.log("Sending summary report prompt to ChatGroq", {"prompt_length": len(prompt)})
        response = self.chatgroq.invoke(prompt)
        report = response.content.strip()
        self.logger.log("Received summary report", {"report_length": len(report)})
        
        #*Extract final score for logging
        try:
            final_score_match = re.search(r'FINAL.*?ASSESSMENT.*?(\d+)(?:/|\s*out\s*of\s*)100', report, re.DOTALL | re.IGNORECASE)
            if final_score_match:
                final_score = float(final_score_match.group(1))
                self.logger.log("Final codebase score", {"score": final_score})
        except Exception as e:
            self.logger.log(f"Could not extract final score: {str(e)}")
        
        self.logger.log("Summary report generation completed")
        return report
    


    def parallelize_first_pass(self, components: List[Dict[str, Any]], component_type: str) -> List[Dict[str, Any]]:
        """
        Analyze components in parallel using ThreadPoolExecutor.
        This speeds up the first pass analysis by processing multiple components concurrently.
        
        Args:
            components: List of components to analyze
            component_type: Type of the components (classes, methods, etc.)
            
        Returns:
            List of analyzed components with first_pass_analysis added
        """
        self.logger.log(f"Starting parallel first pass analysis for {len(components)} {component_type}")
        
        # Use max_workers based on CPU count with a reasonable limit
        # For I/O bound tasks like API calls, we can use more workers than CPU cores
        max_workers = min(32, multiprocessing.cpu_count() * 4)
        
        analyzed_components = []
        
        # Define a helper function to process a single component
        def process_component(comp):
            comp_name = self._get_component_name(comp, component_type)
            self.logger.log(f"Thread analyzing {component_type}: {comp_name}")
            try:
                # Call the existing first_pass_analyze_component method
                analyzed_comp = self.first_pass_analyze_component(comp, component_type)
                return analyzed_comp
            except Exception as e:
                self.logger.log(f"Error in parallel analysis of {comp_name}: {str(e)}")
                # Return the original component if analysis fails
                return comp
        
        # Use ThreadPoolExecutor for parallelization
        # ThreadPoolExecutor is appropriate because the tasks are I/O bound (API calls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all components for processing
            future_to_comp = {executor.submit(process_component, comp): comp for comp in components}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_comp):
                try:
                    analyzed_comp = future.result()
                    analyzed_components.append(analyzed_comp)
                    # Log progress periodically
                    if len(analyzed_components) % 5 == 0:
                        self.logger.log(f"Parallel analysis progress: {len(analyzed_components)}/{len(components)} {component_type} completed")
                except Exception as e:
                    comp = future_to_comp[future]
                    comp_name = self._get_component_name(comp, component_type)
                    self.logger.log(f"Exception in parallel analysis for {comp_name}: {str(e)}")
                    # Add the original component to maintain component count
                    analyzed_components.append(comp)
        
        self.logger.log(f"Completed parallel first pass analysis for {len(analyzed_components)}/{len(components)} {component_type}")
        return analyzed_components
    
    
    #~ FUNCTION TO PROCESS AND ANALYZE JAVA CODE(NO AMBIGUITY, FULLY FUNCTIONAL, 1 ALERT CHANGE REQUIRED)
    def process_and_analyze(self, java_file_path: str):
        """
        Process and analyze Java code with dependencies included in initial extraction.
        First fixes syntax errors, then extracts components, then analyzes them.
        """
        self.logger.log(f"Starting analysis of Java file: {java_file_path}")
        
        try:
            #*Step 1: Fix syntax errors
            self.logger.log("Step 1: Fixing syntax errors...")
            print("Fixing syntax errors...")
            self.syntax_fixer.fix_errors_with_llm(java_file_path)
            self.logger.log("Syntax errors fixed")
            
            #*Step 2: Extract components using JsonExtractor
            self.logger.log("Step 2: Extracting components...")
            print("Extracting components...")
            extracted_components = self.json_extractor.extract_components(java_file_path)
            
            #*Log component counts
            if isinstance(extracted_components, dict):
                component_counts = {k: len(v) for k, v in extracted_components.items() if isinstance(v, list)}
                self.logger.log("Components extracted", {"counts": component_counts})
            else:
                self.logger.log("Components extraction result is not a dictionary", {"type": type(extracted_components)})
            
            #*Convert the extracted components to proper format if needed
            if isinstance(extracted_components, str):
                self.logger.log("Extracted components is a string. Converting to dict...")
                #*Clean up the JSON string
                json_string = extracted_components.replace("```json", "").replace("```", "")
                try: #! how is the string being converted to json using loads instead of ast it wasnt working earlier 
                    extracted_components = json.loads(json_string)
                    self.logger.log("Successfully parsed components using json.loads()")
                except json.JSONDecodeError:
                    self.logger.log("JSONDecodeError. Attempting ast.literal_eval()...")
                    try:
                        extracted_components = ast.literal_eval(json_string)
                        self.logger.log("Successfully parsed components using ast.literal_eval()")
                    except Exception as e:
                        self.logger.log(f"Failed to parse components: {str(e)}")
                        raise

            #*Initialize the analyzed components structure
            analyzed_components = {
                "classes": [],
                "interfaces": [],
                "methods": [],
                "constructors": [],
                "fields": []
            }
            
            #*Store all components for cross-referencing
            self.all_components = extracted_components
            self.logger.log("All components stored for cross-referencing", {
                "keys": list(self.all_components.keys())
            })

            #*Step 3: Analyze components
            self.logger.log("Step 3: Performing first pass analysis in parallel...")
            print("Performing first pass analysis in parallel...")
            for comp_type in analyzed_components.keys():
                components = extracted_components.get(comp_type, [])
                if not components:
                    self.logger.log(f"No {comp_type} to process")
                    continue
                    
                self.logger.log(f"Processing {len(components)} {comp_type} in parallel")
                
                # Use the parallel analysis method
                analyzed_comps = self.parallelize_first_pass(components, comp_type)
                analyzed_components[comp_type] = analyzed_comps
            
            #*Update all_components with first pass results
            self.all_components = analyzed_components
            self.logger.log("First pass analysis completed")
            
            #*Step 4: Second pass analysis with cross-references
            self.logger.log("Step 4: Performing second pass analysis with cross-referencing...")
            print("Performing second pass analysis with cross-referencing...")
            for comp_type in analyzed_components.keys():
                components = analyzed_components[comp_type]
                self.logger.log(f"Processing {len(components)} {comp_type} for second pass")
                
                for i, comp in enumerate(components):
                    #*Second pass analysis using cross-references
                    comp_name = self._get_component_name(comp, comp_type)
                    self.logger.log(f"Second pass for {comp_type}: {comp_name}")
                    analyzed_comp = self.second_pass_analyze_component(comp, comp_type)
                    #*Update the component
                    analyzed_components[comp_type][i] = analyzed_comp
            
            self.logger.log("Second pass analysis completed")
            
            #*Step 5: Final evaluation
            self.logger.log("Step 5: Performing final evaluation...")
            print("Performing final evaluation...")
            for comp_type in analyzed_components.keys():
                components = analyzed_components[comp_type]
                self.logger.log(f"Evaluating {len(components)} {comp_type}")
                
                for i, comp in enumerate(components):
                    #*Final evaluation
                    comp_name = self._get_component_name(comp, comp_type)
                    self.logger.log(f"Evaluating {comp_type}: {comp_name}")
                    evaluated_comp = self.evaluate_component(comp, comp_type)
                    #*Update the component
                    analyzed_components[comp_type][i] = evaluated_comp
            
            self.logger.log("All analysis steps completed successfully")
            return analyzed_components

        except Exception as e:
            self.logger.log(f"Error in process_and_analyze: {str(e)}")
            self.logger.log(f"Traceback: {traceback.format_exc()}")
            print(f"Error in process_and_analyze: {str(e)}")
            print(traceback.format_exc())
            return None

#~ MAIN FUNCTION TO RUN THE ANALYZER(NO AMBIGUITY, FULLY FUNCTIONAL, 0 CHANGES REQUIRED)
if __name__ == "__main__":
    analyzer = JavaCodeAnalyzer(
        model_name="llama-3.3-70b-versatile",
        api_key="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"
    )
    java_file_path = 'dataset/Arrays/Largest product/Solution.java'
    
    #*Process and analyze (this creates the PKL file)
    analyzed_components = analyzer.process_and_analyze(java_file_path)
    
    if analyzed_components:
        #*Generate summary report
        summary_report = analyzer.generate_summary_report(analyzed_components)
        
        #*Clean the result for output
        cleaned_result = analyzer.remove_implementation(analyzed_components)
        
        #*Save to JSON file
        output = {
            "components": cleaned_result,
            "summary_report": summary_report
        }
        
        #*Save detailed output with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'analysis_output_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        #*Also save to standard name for compatibility
        with open('analysis_output.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        analyzer.logger.log("Analysis results saved to files", {
            "files": [output_file, "analysis_output.json"]
        })
        
        #*Print a summary to the console
        print(f"\nAnalysis completed successfully. Results saved to '{output_file}'")
        print("\nSUMMARY REPORT:")
        print("=" * 80)
        print(summary_report)
        print("=" * 80)
        print(f"\nDetailed logs available in: java_analyzer_debug.log")
    else:
        print("Analysis failed to complete successfully. Check java_analyzer_debug.log for details.")