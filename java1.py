code="""package com.example.vehicles;

    import java.util.List;
    import java.util.ArrayList;
    import static java.util.Collections.unmodifiableList;
    import org.springframework.stereotype.Component;

    @FunctionalInterface
    public interface Maintainable {
        void performMaintenance();
        
        default void logMaintenance() {
            System.out.println("Maintenance performed at: " + System.currentTimeMillis());
        }
    }

    public interface Vehicle {
        void start();
        void stop();
        String getType();
    }

    @Component
    public abstract class AbstractVehicle implements Vehicle, Maintainable 
    {
        protected final String manufacturer;
        private volatile boolean running;
        private static final List<String> VALID_TYPES;
        
        static {
            List<String> types = new ArrayList<>();
            types.add("Land");
            types.add("Air");
            types.add("Water");
            VALID_TYPES = unmodifiableList(types);
        }
        
        protected AbstractVehicle(String manufacturer) {
            this.manufacturer = manufacturer;
        }
        
        @Override
        public synchronized void start() {
            running = true;
            System.out.println("Vehicle starting");
        }
        
        @Override
        public void stop() {
            running = false;
            System.out.println("Vehicle stopping");
        }
        
        public static List<String> getValidTypes() {
            return VALID_TYPES;
        }
    }

    @Component
    public class ElectricCar extends AbstractVehicle {
        private transient int batteryLevel;
        @Deprecated
        private String oldField;
        
        public static class Battery {
            private final int capacity;
            
            public Battery(int capacity) {
                this.capacity = capacity;
            }
            
            public int getCapacity() {
                return capacity;
            }
        }
        
        private final Battery battery;
        
        public ElectricCar(String manufacturer, int batteryCapacity) {
            super(manufacturer);
            this.battery = new Battery(batteryCapacity);
            this.batteryLevel = 100;
        }
        
        @Override
        public String getType() {
            return "Land";
        }
        
        @Override
        public void performMaintenance() {
            System.out.println("Performing electric car maintenance");
            batteryLevel = 100;
        }
        
        public synchronized int getBatteryLevel() {
            return batteryLevel;
        }
        
        @Deprecated
        public void oldMethod() throws IllegalStateException {
            throw new IllegalStateException("This method is deprecated");
        }
    }"""