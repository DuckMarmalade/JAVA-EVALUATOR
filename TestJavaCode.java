import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Test class with various Java features to evaluate the JavaCodeExtractor functionality.
 * Includes multiple classes, interfaces, methods, and fields with different modifiers.
 */
public class TestJavaCode {
    // Fields with different modifiers
    private String name;
    private static final int MAX_SIZE = 100;
    protected double value;
    public volatile boolean isActive;
    
    // Static and instance initialization blocks
    static {
        System.out.println("Static initialization block executed");
    }
    
    {
        name = "Default";
        value = 0.0;
        isActive = false;
    }
    
    // Constructors
    public TestJavaCode() {
        System.out.println("Default constructor");
    }
    
    public TestJavaCode(String name) {
        this.name = name;
    }
    
    public TestJavaCode(String name, double value, boolean isActive) {
        this.name = name;
        this.value = value;
        this.isActive = isActive;
    }
    
    // Methods with different modifiers
    public String getName() {
        return name;
    }
    
    protected void setName(String name) {
        this.name = name;
    }
    
    private double calculateValue(double multiplier) {
        return value * multiplier;
    }
    
    public static void printMaxSize() {
        System.out.println("Max size: " + MAX_SIZE);
    }
    
    // Method with generic parameters
    public <T> List<T> processItems(List<T> items) {
        // Some processing
        return new ArrayList<>(items);
    }
    
    // Method with exception handling
    public void riskyOperation() throws Exception {
        if (!isActive) {
            throw new Exception("Operation not allowed when inactive");
        }
        System.out.println("Operation successful");
    }
    
    // Inner class
    public class InnerClass {
        private int innerValue;
        
        public InnerClass(int value) {
            this.innerValue = value;
        }
        
        public void display() {
            System.out.println("Inner class value: " + innerValue);
            System.out.println("Outer class name: " + name);
        }
    }
    
    // Static nested class
    public static class StaticNestedClass {
        private String nestedName;
        
        public StaticNestedClass(String name) {
            this.nestedName = name;
        }
        
        public void displayNested() {
            System.out.println("Nested name: " + nestedName);
            // Note: Cannot access non-static members of outer class
            System.out.println("Outer class max size: " + MAX_SIZE);
        }
    }
    
    // Inner interface
    public interface TestInterface {
        void execute();
        
        default void defaultMethod() {
            System.out.println("Default method in interface");
        }
    }
    
    // Implementation of inner interface
    public class InterfaceImplementation implements TestInterface {
        @Override
        public void execute() {
            System.out.println("Executing interface method");
        }
        
        // Using outer class method
        public void useOuterMethod() {
            setName("Updated from inner class");
        }
    }
}

// Additional class in the same file
class SecondaryClass extends TestJavaCode {
    private int secondaryValue;
    
    public SecondaryClass() {
        super("Secondary");
        this.secondaryValue = 42;
    }
    
    @Override
    public String getName() {
        return super.getName() + " (Secondary)";
    }
    
    public void updateValue(double newValue) {
        value = newValue;  // Accessing protected field from parent
    }
}

// Interface with various methods
interface DataProcessor {
    void process(String data);
    
    List<String> filter(List<String> items, String criteria);
    
    default Map<String, Object> createMetadata() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("processor", "Basic");
        metadata.put("version", 1.0);
        return metadata;
    }
}

// Implementation of the interface
class BasicDataProcessor implements DataProcessor {
    @Override
    public void process(String data) {
        System.out.println("Processing: " + data);
    }
    
    @Override
    public List<String> filter(List<String> items, String criteria) {
        List<String> result = new ArrayList<>();
        for (String item : items) {
            if (item.contains(criteria)) {
                result.add(item);
            }
        }
        return result;
    }
    
    // Custom method
    public void clearCache() {
        System.out.println("Cache cleared");
    }
}