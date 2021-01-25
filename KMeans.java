import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Represents an implementation of the k-means clustering algorithm
 */
public class KMeans {
  CSVReader data;
  int numClusters;
  HashMap<Integer, float[]>centroidMap;

  ArrayList<float[]>[] clusters;
  ArrayList<String>[] targetNameClusters;

  /**
   * Creates instance of k-means algorithm. numClusters specifies the number
   * of clusters we want to train the algorithm on, and the data we will be
   * using is also specified.
   */
  public KMeans(int numClusters, CSVReader data) {
    this.data = data;
    this.numClusters = numClusters;

    // Clusters that contain the X values
    clusters = new ArrayList[numClusters];

    // Clusters that contain the y values
    targetNameClusters = new ArrayList[numClusters];

    initialize();
  }

  /**
   * Initializes the centroids for each cluster. Randomly initializes first
   * centroid, then initializes the next centroid by setting it equal to the point
   * in our dataset farthest away from the previously initialized centroid.
   */
  public void initialize() {
    Random r = new Random();
    int columns = data.features.length;
    float[] centroid = new float[columns];
    centroidMap = new HashMap<>();

    // Initialize clusters
    for(int i = 0; i < numClusters; i++ ) {
      clusters[i] = new ArrayList<>();
      targetNameClusters[i] = new ArrayList<>();
    }

    // Randomly initialize first cluster's centroid
    for (int i = 0; i <columns; i++){
      centroid[i] = r.nextFloat();
    }centroidMap.put(0, centroid);

    // Iterate remaining clusters
    for(int i = 1; i < numClusters; i++) {
      float[] distances = new float[data.rows];

      // Iterate each row and get distance between row and previous centroid
      for(int k = 0; k < data.rows; k++){
        distances[k] = euclideanDistance( data.X[k], centroidMap.get(i-1) );
      }

      int maxIndex = 0;
      float maxValue = Float.MIN_VALUE;

      // Find point with max distance from previous centroid
      for( int d =0; d < distances.length; d++) {
        if(distances[d] > maxValue) {
          maxIndex = d;
          maxValue = distances[d];
        }
      }
      // Set next centroid to point with furtherst distance from previous centroid
      float[] nextCentroid = data.X[maxIndex].clone();
      centroidMap.put(i, nextCentroid);
    }
  }

  /**
   * Fits our model. This is done by iterating all points and adding the
   * point to a cluster which contains the closest centroid. We then update our
   * centroid to a new optimal position
   */
  public void fit() {
    float min ;
    int minCluster;

    for(int i =0 ; i<numClusters; i++){
      this.clusters[i] = new ArrayList<>();
      this.targetNameClusters[i] = new ArrayList<>();
    }

    // Iterate all rows
    for (int r = 0; r < data.rows; r++){
      minCluster = -1;
      min = Float.MAX_VALUE;

      // Iterate each cluster and find centroid with min distance to row
      for(int cluster = 0; cluster < numClusters; cluster++) {
        float distance = euclideanDistance(this.data.X[r], this.centroidMap.get(cluster));

        if(distance < min) {
          min = distance;
          minCluster = cluster;
        }
      }

      // Add the row to cluster with the closest centroid
      this.clusters[minCluster].add(data.X[r]);
      this.targetNameClusters[minCluster].add(data.y[r]);
    }
    updateCentroids();
  }

  /**
   * Update our centroid to a new optimal position. This is done by setting the
   * centroid equal to the average of all the points within that cluster.
   */
  public void updateCentroids() {
    float[] sum;
    float[] meanCentroid;
    HashMap<Integer, float[]> newCentroidList = new HashMap<>();

    // Iterate clusters
    for (int cluster = 0; cluster < numClusters; cluster++) {
      sum = new float[data.columns];
      meanCentroid = new float[data.columns];

      for (int i = 0; i < data.columns; i++) {
        sum[i] = 0f;
        meanCentroid[i] = 0f;
      }

      // Iterate points in current cluster and sum up their columns
      for (int point = 0; point < clusters[cluster].size(); point++) {
        for (int column = 0; column < data.columns; column++) {
          sum[column] += clusters[cluster].get(point)[column];
        }
      }

      // Calculate new centroid by taking the mean of all points in cluster
      for (int c = 0; c < sum.length; c++) {
        meanCentroid[c] = sum[c] / clusters[cluster].size();
      }
      newCentroidList.put(cluster, meanCentroid);
    }
    this.centroidMap = newCentroidList;
  }

  /**
   * Calculates euclidean distance between two points.
   * @param x1 Point in our dataset
   * @param x2 Point in our dataset
   */
  public float euclideanDistance(float [] x1, float [] x2) {
    double output = 0.0;
    for (int i = 0 ; i<x1.length; i++){
      output += Math.pow((x1[i] - x2[i]), 2);
    }
    return (float) Math.sqrt(output);
  }

  /**
   * Calculates manhattan distance between two points.
   * @param x1 Point in our dataset
   * @param x2 Point in our dataset
   */
  public float manhattanDistance(float[] x1, float[] x2) {
    double output = 0.0;
    for(int i = 0; i < x1.length; i++){
      output += (Math.abs(x1[i]-x2[i]));
    }
    return (float) output;
  }
}
