import java.util.HashMap;
import java.util.Map;

/**
 * Program driver. Fits kmeans to data for 10 iterations.
 * When complete, iterates through each cluster and outputs
 * what labels are within that cluster.
 */
public class Main {
  public static void main(String[] args) {
    // Preprocess data
    CSVReader reader = new CSVReader("Iris.csv");
    reader.parseData();
    reader.minMaxScaler();

    KMeans kmeans = new KMeans(3, reader);
    for (int i = 0; i < 10; i++) { // Fitting loop
      kmeans.fit();
    }

    // Print labels and count within each cluster
    for (int i = 0; i < kmeans.targetNameClusters.length; i++) {
      Map<String, Integer> labelCounts = new HashMap<>();
      System.out.println("====================================");
      System.out.println("Cluster" + (i + 1) + " target values");

      for (String label : kmeans.targetNameClusters[i]) {
        if (labelCounts.containsKey(label)) {
          labelCounts.put(label, labelCounts.get(label) + 1);
        } else {
          labelCounts.put(label, 1);
        }
      }

      for (Map.Entry<String, Integer> entry : labelCounts.entrySet()) {
        System.out.println( entry.getKey() + ": " + entry.getValue());
      }
    }
  }
}
