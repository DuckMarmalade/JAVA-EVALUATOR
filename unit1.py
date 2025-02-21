import subprocess
import re
import json
from langchain_groq import ChatGroq

class JavaSyntaxFixer:
    def __init__(self, model_name, api_key, temperature=0):
        self.chatgroq = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )

    def check_java_syntax(self, java_file_path):
        """Compiles the Java file and returns error details."""
        error_dict = {}
        try:
            result = subprocess.run(["javac", java_file_path], capture_output=True, text=True)
            
            if result.stderr:
                errors = result.stderr.strip().split('\n')
                for error in errors:
                    if 'error' in error and not re.search(r'\d+ errors?$', error):
                        line_match = re.search(r':(\d+):', error)
                        if line_match:
                            line_num = int(line_match.group(1))
                            error_msg = re.sub(r'^.*?:\d+:', '', error).strip()
                            error_dict.setdefault(line_num, []).append(error_msg)
                
                last_line = errors[-1]
                match = re.search(r'(\d+) errors?', last_line)
                error_count = int(match.group(1)) if match else len(error_dict)
            else:
                error_count = 0
        
            return error_count, error_dict
        except Exception as e:
            print(f"Compilation failed: {e}")
            return -1, {}

    def get_java_code(self, java_file_path):
        """Reads and returns the Java file content."""
        with open(java_file_path, "r") as file:
            return file.read()

    def update_java_file(self, java_file_path, new_code):
        """Writes the corrected Java code back to the file."""
        with open(java_file_path, "w") as file:
            file.write(new_code)

    def strip_code_markers(self, response):
        """Strips code markers and unnecessary text from the LLM response."""
        cleaned = re.sub(r'^.*?```java\n', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'^java\n', '', cleaned)
        return cleaned.strip()

    def fix_errors_with_llm(self, java_file_path):
        """Iteratively fixes Java syntax errors using the LLM."""
        while True:
            error_count, error_dict = self.check_java_syntax(java_file_path)
            if error_count == 0:
                print("No syntax errors remaining!")
                break
            
            java_code = self.get_java_code(java_file_path)
            line_number, error_messages = next(iter(error_dict.items()))
            error_str = " | ".join(error_messages)
            print(error_dict.items())
            print(f"Line number: {line_number}, error messages: {error_str}")
            # stupid llm doesnt fix one error at a time so we have to do this
            prompt = (
                "You are a Java syntax error fixer.\n"
                "Your task is to correct only the error on the specified line without modifying any other part of the code.\n\n"
                f"Error at line {line_number}: {error_str}\n\n"
                "Below is the complete current code:\n"
                f"{java_code}\n\n"
                "Instructions:\n"
                "1. Modify only the line at line number " + str(line_number) + " to fix the error.\n"
                "2. Do not modify any other lines, even if they contain errors.\n"
                "3. Return the complete code with only that one line modified.\n"
                "4. Do not include any additional comments or changes."
            )
            
            response = self.chatgroq.invoke(prompt).content
            if response:
                
                cleaned_response = self.strip_code_markers(response)
                self.update_java_file(java_file_path, cleaned_response)
                print(f"Fixed error on line {line_number}. Recompiling...")
            else:
                print("LLM failed to provide a correction. Exiting...")
                break
