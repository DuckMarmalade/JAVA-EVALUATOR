from langchain_groq import ChatGroq
import json

class JavaCodeExtractor:
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.2):
        self.chatgroq = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )
        # self.syntax_fixer = JavaSyntaxFixer(model_name, api_key)  # Initialize JavaSyntaxFixer

    def extract_components(self, java_file_path: str):
        # First, fix any syntax errors in the Java code before extraction
        # self.syntax_fixer.fix_errors_with_llm(java_file_path)
        
        #* Read the fixed Java code
        with open(java_file_path, "r") as file:
            java_code = file.read()
        #^ enums yet to be added . to do later 
        #^ Null is produced ometimes instead of empty list for fields that are empty. this leads to errors. 
        prompt = f"""
        Analyze the following Java code and extract its components in a structured format with their dependencies. 
        Do not truncate body of classes, interfaces and methods.
        Return ONLY the JSON object with no additional text, comments, or explanations.
        Identify all syntax errors and do not fix any syntax errors.
        
        For each component, provide the following details including dependencies:

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

        Imports:
        - Package name
        - Static imports
        - Dependencies:
          - Used by which classes/methods

        Java Code:
        {java_code}

        Return this JSON structure with dependencies included in each component:
        {{
            "classes": [{{
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
                "dependencies": {{
                    "inherited_methods": ["method_name"],
                    "overridden_methods": ["method_name"],
                    "interface_implementations": ["method_name"],
                    "field_dependencies": [{{"field": "field_name", "used_by": ["method_name"]}}],
                    "method_dependencies": [{{"method": "method_name", "calls": ["method_name"]}}]
                }},
                "description": ""
            }}],
            "interfaces": [{{
                "interface_name": "",
                "modifiers": ["modifier"],
                "methods": ["method_name"],
                "default_methods": ["default_method_name"],
                "enclosing_class": ["enclosing_class_name"],
                "implementation": "",
                "dependencies": {{
                    "implementing_classes": ["class_name"],
                    "extended_interfaces": ["interface_name"]
                }},
                "description": ""
            }}],
            "methods": [{{
                "method_name": "",
                "modifiers": ["modifier"],
                "return_type": "",
                "parameters": [{{"name": "", "type": ""}}],
                "throws": ["exception_type"],
                "enclosing_class": ["enclosing_class_name"],
                "implementation": "",
                "dependencies": {{
                    "called_methods": ["method_name"],
                    "used_fields": ["field_name"],
                    "overrides": "superclass_method_name",
                    "implements": "interface_method_name"
                }},
                "description": ""
            }}],
            "constructors": [{{
                "constructor_name": "",
                "modifiers": ["modifier"],
                "parameters": [{{"name": "", "type": ""}}],
                "throws": ["exception_type"],
                "enclosing_class": ["enclosing_class_name"],
                "implementation": "",
                "dependencies": {{
                    "constructor_chaining": "constructor_name",
                    "super_constructor": true,
                    "initialized_fields": ["field_name"],
                    "called_methods": ["method_name"]
                }},
                "description": ""
            }}],
            "fields": [{{
                "field_name": "",
                "type": "",
                "modifiers": ["modifier"],
                "enclosing_class": ["enclosing_class_name"],
                "initial_value": "",
                "implementation": "",
                "dependencies": {{
                    "used_by_methods": ["method_name"],
                    "shadows_field": "superclass_field_name",
                    "initialization_dependencies": ["field_or_method_name"]
                }},
                "description": ""
            }}],
            "imports": [{{
                "standard_imports": ["package_name"],
                "static_imports": ["static_import"],
                "dependencies": {{
                    "used_by": ["class_or_method_name"]
                }}
            }}]
        }}"""
        response = self.chatgroq.invoke(prompt)
        #*Convert string response to JSON
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            #*If response is not valid JSON, return raw content
            return response.content
    