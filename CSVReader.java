import java.io.File;
import java.io.FileNotFoundException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * CSVReader pre-processes iris.csv so k-means clustering
 * can be applied
 */
public class CSVReader {
  String filename;
  String [] features;
  float [][] X;
  String [] y;

  int columns;
  int rows;

  public CSVReader(String fn) {
    this.filename = fn;
  }

  /**
   * Read in csv file and store attributes in the 2d array X, and labels are
   * stored in 1d array y. Save features of dataset and num of rows/cols in
   * class fields
   */
  public void parseData() {
    ArrayList<float[]> xData = new ArrayList<>();
    ArrayList<String> yData = new ArrayList<>();
    URL url = getClass().getResource(filename);

    try {
      Scanner scanner = new Scanner(new File(url.getPath())).useDelimiter("\n");
      String [] tempFeatures = scanner.nextLine().split(",");

      // Remove ID and target name from feature list
      this.features = Arrays.copyOfRange(tempFeatures, 1, tempFeatures.length-1);

      // Save feature values and target names
      while(scanner.hasNext()) {
        String [] attributes = scanner.nextLine().split(",");
        float [] curX = new float[attributes.length-2];
        String targetValue = attributes[attributes.length-1];

        for( int i = 1; i < attributes.length-1; i++ ){
          curX[i-1] = Float.parseFloat(attributes[i]);
        }

        xData.add(curX);
        yData.add(targetValue);
      }

      this.columns = this.features.length;
      this.rows = xData.size();
      this.X = xData.toArray(new float[rows][columns]);
      this.y = yData.toArray(new String[rows]);

    }
    catch(FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  /**
   * Apply min/max scaling to the X field.
   * Scaling formula is (x - x_min)/(x_max - x_min)
   */
  public void minMaxScaler() {
    float[] min = new float[columns];
    float[] max = new float[columns];

    // Initialize min/max values
    for(int i = 0; i < columns; i++){
      min[i] = Float.MAX_VALUE;
      max[i] = Float.MIN_VALUE;
    }

    // Iterate all data and store min/max values for each column
    for(int j = 0; j < rows; j++) {
      for(int k = 0; k< columns; k++){
        if (X[j][k] < min[k] ) {
          min[k] = X[j][k];
        }
        else if(X[j][k] > max[k]) {
          max[k] = X[j][k];
        }
      }
    }

    // Apply min/max normalization to each index in X
    for(int j = 0; j < rows; j++) {
      for (int k = 0; k < columns; k++) {
        X[j][k] = (X[j][k] - min[k]) / (max[k]-min[k]);
      }
    }
  }

  /**
   * Apply standard scaling to the X field.
   * Scaling formula is (x - mean) / stdev
   */
  public void standardScaler() {
    float[] mean = new float[columns];
    float[] stdev = new float[columns];

    for(int i = 0; i < columns; i++){
      mean[i] = 0.0f;
      stdev[i] = 0.0f;
    }

    // Compute mean
    for(int j = 0; j < rows; j++) {
      for (int k = 0; k < columns; k++) {
        mean[k] += X[j][k];
      }
    }
    for(int j = 0; j < columns; j++) {
      mean[j] /= X.length;
    }

    // Start computing stdev of each column
    for(int j = 0; j < rows; j++) {
      for (int k = 0; k < columns; k++) {
        stdev[k] += Math.pow(X[j][k] - mean[k], 2);
      }
    }

    // Complete remaining stdev calculations
    for(int j = 0; j < columns; j++) {
      stdev[j] /= X.length;
      stdev[j] = (float) Math.sqrt(stdev[j]);
    }

    // Apply standard normalization to each index in X
    for(int j = 0; j < rows; j++) {
      for (int k = 0; k < columns; k++) {
        X[j][k] = (X[j][k] - mean[k]) / (stdev[k]);
      }
    }
  }
}
