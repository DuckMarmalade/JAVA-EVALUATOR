from langchain_groq import ChatGroq
import json
from typing import Dict, List, Any, Optional, Union
import re

class JavaCodeExtractor:
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.2):
        """
        Initialize the JavaCodeExtractor with specified LLM configuration.
        
        Args:
            model_name: Name of the Groq model to use
            api_key: API key for accessing Groq API
            temperature: Temperature parameter for LLM generation (0.0-1.0)
        """
        self.chatgroq = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )
        
    def extract_components(self, java_file_path: str) -> Dict[str, Any]:
        """
        Extract components from a Java file and return them in a structured format.
        
        Args:
            java_file_path: Path to the Java file to analyze
            
        Returns:
            Dictionary containing structured Java components or error information
        """
        try:
            # Read the Java code
            with open(java_file_path, "r") as file:
                java_code = file.read()
                
            # Extract components using the LLM
            prompt = self._build_extraction_prompt(java_code)
            response = self.chatgroq.invoke(prompt)
            
            # Process and validate the response
            extracted_data = self._process_llm_response(response.content)
            if extracted_data and self._validate_extracted_data(extracted_data):
                # Clean any potential null values in lists
                cleaned_data = self._clean_null_lists(extracted_data)
                return cleaned_data
            else:
                return {
                    "error": "Failed to extract valid component data",
                    "classes": [],
                    "interfaces": [],
                    "methods": [],
                    "constructors": [],
                    "fields": [],
                    "imports": []
                }
                
        except Exception as e:
            # Return structured error response
            return {
                "error": f"Exception during component extraction: {str(e)}",
                "classes": [],
                "interfaces": [],
                "methods": [],
                "constructors": [],
                "fields": [],
                "imports": []
            }
    
    def _clean_null_lists(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively clean the data to replace lists containing only null with empty lists.
        
        Args:
            data: The data structure to clean
            
        Returns:
            The cleaned data structure with proper empty lists
        """
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._clean_null_lists(value)
        elif isinstance(data, list):
            # If list contains only None/null values, replace with empty list
            if all(item is None for item in data):
                return []
            # Otherwise process each item in the list
            return [self._clean_null_lists(item) if item is not None else None for item in data]
            # Filter out None values from lists
            data = [item for item in data if item is not None]
        return data
    
    def _build_extraction_prompt(self, java_code: str) -> str:
        """
        Build a structured prompt for extracting Java components.
        
        Args:
            java_code: Java code to analyze
            
        Returns:
            Formatted prompt string for the LLM
        """
        # Base instruction
        base_instruction = """
        Analyze the following Java code and extract its components in a structured format with their dependencies. 
        Do not truncate body of classes, interfaces and methods.
        Return ONLY the JSON object with no additional text, comments, or explanations.
        Identify all syntax errors and do not fix any syntax errors.
        IMPORTANT: For any empty lists, return [] and not [null]. Never include null values in lists.
        """
        
        # Component specifications
        class_spec = """
        Classes(including nested classes):
        - Name
        - Modifiers (e.g., public, private, abstract, final, strictfp)
        - Fields (name only)
        - Methods (name only)
        - Constructors (name only)
        - Enclosing Class (name only)
        - Nested Classes or Interfaces (name only)
        - Super Class (name only)
        - Implemented Interfaces (name only)
        - Sub Class (name only)
        - Implementation (Body of the Class, with newline characters escaped as \\n)
        - Dependencies:
          - Inherited methods (list methods inherited from superclass)
          - Overridden methods (list methods that override superclass)
          - Interface implementations (list methods implementing interfaces)
          - Field dependencies (fields used by which methods)
          - Method dependencies (methods called by other methods)
        - Brief description
        """
        
        interface_spec = """
        Interfaces:
        - Name
        - Enclosing class (name only)
        - Modifiers (e.g., public, private, abstract, static)
        - Methods (name only)
        - Default methods (name only)
        - Implementation (Body of the interface, with newline characters escaped as \\n)
        - Dependencies:
          - Implementing classes (list of classes implementing this interface)
          - Extended interfaces (list of interfaces this interface extends)
        - Brief description
        """
        
        method_spec = """
        Methods:
        - Name
        - Enclosing class/interface (name only)
        - Modifiers (e.g., public, private, static, final, synchronized, native, abstract)
        - Return type
        - Parameters (with types)
        - Throws declarations
        - Implementation (Body of the method, with newline characters escaped as \\n)
        - Dependencies:
          - Called methods (list of methods this method calls)
          - Used fields (list of fields this method uses)
          - Overridden method (if this method overrides a superclass method)
          - Interface implementation (if this method implements an interface method)
        - Brief description
        """
        
        constructor_spec = """
        Constructors:
        - Name
        - Enclosing class (name only)
        - Modifiers (e.g., public, private, protected)
        - Parameters (with types)
        - Throws declarations
        - Implementation (Body of the constructor, with newline characters escaped as \\n)
        - Dependencies:
          - Constructor chaining (this() calls)
          - Super constructor calls
          - Field initializations
          - Method calls
        - Brief description
        """
        
        field_spec = """
        Fields:
        - Name
        - Enclosing class
        - Type
        - Modifiers (e.g., public, private, static, final, volatile, transient)
        - Initial value
        - Implementation (Field declaration line, with newline characters escaped as \\n)
        - Dependencies:
          - Methods using this field
          - Shadowed fields (if this field shadows a superclass field)
          - Initialization dependencies (other fields/methods used in initialization)
        - Usage context
        """
        
        import_spec = """
        Imports:
        - Package name
        - Static imports
        - Dependencies:
          - Used by which classes/methods
        """
        
        # JSON structure template
        json_structure = """
        {
            "classes": [{
                "class_name": "",
                "modifiers": ["modifier"],
                "fields": ["field_name"],
                "methods": ["method_name"],
                "constructors": ["constructor_name"],
                "enclosing_class": ["enclosing_class_name"],
                "nested_classes_or_interfaces": ["nested_name"],
                "super_class": ["super_class_name"],
                "implemented_interfaces": ["interface_name"],
                "sub_class": ["sub_class_name"],
                "implementation": "",
                "dependencies": {
                    "inherited_methods": ["method_name"],
                    "overridden_methods": ["method_name"],
                    "interface_implementations": ["method_name"],
                    "field_dependencies": [{"field": "field_name", "used_by": ["method_name"]}],
                    "method_dependencies": [{"method": "method_name", "calls": ["method_name"]}]
                },
                "description": ""
            }],
            "interfaces": [{
                "interface_name": "",
                "modifiers": ["modifier"],
                "methods": ["method_name"],
                "default_methods": ["default_method_name"],
                "enclosing_class": ["enclosing_class_name"],
                "implementation": "",
                "dependencies": {
                    "implementing_classes": ["class_name"],
                    "extended_interfaces": ["interface_name"]
                },
                "description": ""
            }],
            "methods": [{
                "method_name": "",
                "modifiers": ["modifier"],
                "return_type": "",
                "parameters": [{"name": "", "type": ""}],
                "throws": ["exception_type"],
                "enclosing_class": ["enclosing_class_name"],
                "implementation": "",
                "dependencies": {
                    "called_methods": ["method_name"],
                    "used_fields": ["field_name"],
                    "overrides": "superclass_method_name",
                    "implements": "interface_method_name"
                },
                "description": ""
            }],
            "constructors": [{
                "constructor_name": "",
                "modifiers": ["modifier"],
                "parameters": [{"name": "", "type": ""}],
                "throws": ["exception_type"],
                "enclosing_class": ["enclosing_class_name"],
                "implementation": "",
                "dependencies": {
                    "constructor_chaining": "constructor_name",
                    "super_constructor": true,
                    "initialized_fields": ["field_name"],
                    "called_methods": ["method_name"]
                },
                "description": ""
            }],
            "fields": [{
                "field_name": "",
                "type": "",
                "modifiers": ["modifier"],
                "enclosing_class": ["enclosing_class_name"],
                "initial_value": "",
                "implementation": "",
                "dependencies": {
                    "used_by_methods": ["method_name"],
                    "shadows_field": "superclass_field_name",
                    "initialization_dependencies": ["field_or_method_name"]
                },
                "description": ""
            }],
            "imports": [{
                "standard_imports": ["package_name"],
                "static_imports": ["static_import"],
                "dependencies": {
                    "used_by": ["class_or_method_name"]
                }
            }]
        }
        """
        
        # Combine all sections into a final prompt
        prompt = f"{base_instruction}\n\nFor each component, provide the following details including dependencies:\n\n{class_spec}\n\n{interface_spec}\n\n{method_spec}\n\n{constructor_spec}\n\n{field_spec}\n\n{import_spec}\n\nJava Code:\n{java_code}\n\nReturn this JSON structure with dependencies included in each component:\n{json_structure}"
        
        return prompt
    
    def _process_llm_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Process and extract JSON data from LLM response.
        
        Args:
            response_content: Raw content from LLM response
            
        Returns:
            Parsed JSON data or None if parsing fails
        """
        # Try to extract JSON from the response
        try:
            # First, try direct JSON parsing
            parsed_data = json.loads(response_content)
            return parsed_data
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown code blocks
            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            matches = re.findall(json_pattern, response_content)
            
            if matches:
                # Try parsing each match until one works
                for match in matches:
                    try:
                        parsed_data = json.loads(match)
                        return parsed_data
                    except json.JSONDecodeError:
                        continue
            
            # If no code blocks found, try removing leading/trailing text
            # This handles cases where the LLM adds explanatory text
            cleaned_content = re.sub(r'^[^{]*', '', response_content)
            cleaned_content = re.sub(r'[^}]*$', '', cleaned_content)
            
            try:
                parsed_data = json.loads(cleaned_content)
                return parsed_data
            except json.JSONDecodeError:
                # Return None if all parsing attempts fail
                return None
    
    def _validate_extracted_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the extracted data has the expected structure.
        
        Args:
            data: Parsed JSON data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check that required top-level fields exist
        required_fields = ["classes", "interfaces", "methods", "constructors", "fields"]
        for field in required_fields:
            if field not in data or not isinstance(data[field], list):
                return False
                
        # Basic validation for each component type
        if data["classes"]:
            for cls in data["classes"]:
                if "class_name" not in cls or not isinstance(cls.get("dependencies", {}), dict):
                    return False
                    
        if data["methods"]:
            for method in data["methods"]:
                if "method_name" not in method:
                    return False
                    
        if data["interfaces"]:
            for interface in data["interfaces"]:
                if "interface_name" not in interface:
                    return False
                    
        if data["constructors"]:
            for constructor in data["constructors"]:
                if "constructor_name" not in constructor:
                    return False
                    
        if data["fields"]:
            for field in data["fields"]:
                if "field_name" not in field:
                    return False
                    
        # All validations passed
        return True
    
if __name__ == "__main__":
    extractor = JavaCodeExtractor(
        model_name="llama-3.3-70b-versatile",
        api_key="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"
    )
    java_file_path = "TestJavaCode.java"  # Point to your test file
    result = extractor.extract_components(java_file_path)
    
    # Save the extracted components to a JSON file
    with open("extracted_components.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Components successfully extracted and saved to extracted_components.json")