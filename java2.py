sort="""import java.io.*;
import java.util.*;

class Playr {
    private String playerName;
    private Roleangle role;
    private int runsScored;
    private int wicketsTaken;
    private String teamName;

    public Playr(String playerName, Roleangle role, int runsScored, int wicketsTaken, String teamName) {
        this.playerName = playerName;
        this.role = role;
        this.runsScored = runsScored;
        this.wicketsTaken = wicketsTaken;
        this.teamName = teamName;
    }

    public String getPlayerName() {
        return playerName;
    }

    public void setPlayerName(String playerName) {
        this.playerName = playerName;
    }

    public Roleangle getRole() {
        return role;
    }

    public void setRole(Roleangle role) {
        this.role = role;
    }

    public int getRunsScored() {
        return runsScored;
    }

    public void setRunsScored(int runsScored) {
        this.runsScored = runsScored;
    }

    public int getWicketsTaken() {
        return wicketsTaken;
    }

    public void setWicketsTaken(int wicketsTaken) {
        this.wicketsTaken = wicketsTaken;
    }

    public String getTeamName() {
        return teamName;
    }

    public void setTeamName(String teamName) {
        this.teamName = teamName;
    }

    @Override
    public String toString() {
        return "Player{" +
               "playerName='" + playerName + '\'' +
               ", role=" + role +
               ", runsScored=" + runsScored +
               ", wicketsTaken=" + wicketsTaken +
               ", teamName='" + teamName + '\'' +
               '}';
    }

    public String toCsvFormat() {
        return String.format("%s,%s,%d,%d,%s",
                playerName, role, runsScored, wicketsTaken, teamName);
    }
}

enum Roleangle {
    BATSMAN, BOWLER, ALL_ROUNDER;

    public static Roleangle fromString(String role) {
        switch (role.toUpperCase().replace("-", "_")) {
            case "BATSMAN":
                return BATSMAN;
            case "BOWLER":
                return BOWLER;
            case "ALL_ROUNDER":
                return ALL_ROUNDER;
            default:
                throw new IllegalArgumentException("Unknown role: " + role);
        }
    }
}

class RunsComparator implements Comparator<Playr> {
    @Override
    public int compare(Playr p1, Playr p2) {
        return Integer.compare(p2.getRunsScored(), p1.getRunsScored());
    }
}

class CricketDataHandler {
    public List<Playr> readPlayersFromFile(String fileName) throws IOException {
        List<Playr> players = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        br.readLine();
        String line;
        while ((line = br.readLine()) != null) {
            String[] data = line.split(",");
            Playr player = new Playr(
                    data[0], 
                    Roleangle.fromString(data[1]), 
                    Integer.parseInt(data[2]), 
                    Integer.parseInt(data[3]), 
                    data[4]
            );
            players.add(player);
        }
        br.close();
        return players;
    }

    public void writePlayersToFile(String fileName, List<Playr> players) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
        bw.write("PlayerName,Role,RunsScored,WicketsTaken,TeamName\n");
        for (Playr player : players) {
            bw.write(player.toCsvFormat() + "\n");
        }
        bw.close();
    }

    public void updateePlayerStats(List<Playr> players, String playerName, int runs, int wickets) {
        for (Playr player : players) {
            if (player.getPlayerName().equalsIgnoreCase(playerName)) {
                player.setRunsScored(player.getRunsScored() + runs);
                player.setWicketsTaken(player.getWicketsTaken() + wickets);
                System.out.println("Updated statistics for " + playerName);
                return;
            }
        }
        throw new IllegalArgumentException("Player not found: " + playerName);
    }

    public double calculateTeamAverageRuns(List<Playr> players, String teamName) {
        PlayerFilter<String> teamFilter = new TeamFilterStrategy();
        List<Playr> teamPlayers = teamFilter.filter(players, teamName);
        if (teamPlayers.isEmpty()) {
            throw new IllegalArgumentException("Team not found.");
        }
        int totalRuns = 0;
        for (Playr player : teamPlayers) {
            totalRuns += player.getRunsScored();
        }
        return (double) totalRuns / teamPlayers.size();
    }
}

@FunctionalInterface
interface PlayerFilter<T> {
    List<Playr> filter(List<Playr> players, T value);
}

class TeamFilterStrategy implements PlayerFilter<String> {
    
    @Override
    public List<Playr> filter(List<Playr> players, String teamName) {
        List<Playr> filteredPlayers = new ArrayList<>();
        for (Playr player : players) {
            if (player.getTeamName().equalsIgnoreCase(teamName)) {
                filteredPlayers.add(player);
            }
        }
        return filteredPlayers;
    }
}

class AllRounderStatsFilter implements PlayerFilter<int[]> {
    
    @Override
    public List<Playr> filter(List<Playr> players, int[] criteria) {
        List<Playr> filteredPlayers = new ArrayList<>();
        for (Playr player : players) {
            if (player.getRole() == Roleangle.ALL_ROUNDER 
                && player.getRunsScored() > criteria[0] 
                && player.getWicketsTaken() > criteria[1]) {
                filteredPlayers.add(player);
            }
        }
        return filteredPlayers;
    }
}

public class CricketAnalyticsSolution {
    private static void printPlayers(String header, List<Playr> players) {
        System.out.println("\n--- " + header + " ---");
        for (Playr player : players) {
            System.out.println(player);
        }
    }

    public static void main(String[] args) {
        CricketDataHandler dataHandler = new CricketDataHandler();
        List<Playr> players = new ArrayList<>();

        try {
            players = dataHandler.readPlayersFromFile("inputCricketData.csv");
        } catch (FileNotFoundException e) {
            System.out.println("Error: File not found.");
            return;
        } catch (IOException e) {
            System.out.println("Error: Unable to read file.");
            return;
        }

        PlayerFilter<String> teamFilter = new TeamFilterStrategy();
        List<Playr> indianPlayers = teamFilter.filter(players, "India");
        printPlayers("Players from India", indianPlayers);

        List<Playr> australianPlayers = teamFilter.filter(players, "Australia");
        printPlayers("Players from Australia", australianPlayers);

        System.out.println("\n--- Updating Player Statistics ---");
        dataHandler.updateePlayerStats(players, "Virat Kohli", 82, 0);
        dataHandler.updateePlayerStats(players, "Jasprit Bumrah", 2, 3);
        dataHandler.updateePlayerStats(players, "Steve Smith", 144, 0);
        dataHandler.updateePlayerStats(players, "Pat Cummins", 12, 4);

        players.sort(new RunsComparator());
        printPlayers("Players Sorted by Runs", players);

        System.out.println("\n--- Team Averages ---");
       double indiaAvg = dataHandler.calculateTeamAverageRuns(players, "India");
        System.out.println("Average Runs for Team India: " + indiaAvg);

        double ausAvg = dataHandler.calculateTeamAverageRuns(players, "Australia");
        System.out.println("Average Runs for Team Australia: " + ausAvg);

        double engAvg = dataHandler.calculateTeamAverageRuns(players, "England");
        System.out.println("Average Runs for Team England: " + engAvg);

        int[] criteria = {2000, 100}; 
        List<Playr> goodAllRounders = new AllRounderStatsFilter().filter(players, criteria);
        printPlayers("All-rounders with good stats (>2000 runs and >100 wickets)", goodAllRounders);

        try {
            dataHandler.writePlayersToFile("outputCricketData.csv", players);
        } catch (IOException e) {
            System.out.println("Error: Unable to write to file.");
        }
    }
}"""