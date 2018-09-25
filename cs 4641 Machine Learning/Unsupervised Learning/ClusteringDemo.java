import weka.core.Instances;
import weka.clusterers.DensityBasedClusterer;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.ClusterEvaluation;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.Filter;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.core.converters.CSVSaver;

import java.util.Random;
import java.io.File;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.FileReader;

/**
 * An example class that shows the use of Weka clusterers from Java.
 *
 * @author  FracPete
 */

class ClusterThread extends Thread {

  private String dataset;
  private String filename;

  public ClusterThread(String dataset, String filename)
  {
        this.dataset = dataset;
        this.filename = filename;
  }

  /**
   * Run clusterers
   *
   * @param filename      the name of the ARFF file to run on
   */
  public void run() {
    try
    {
    ClusterEvaluation eval;
    Instances               data;
    String[]                options;
    DensityBasedClusterer   cl;

    // Load data
    data = new Instances(new BufferedReader(new FileReader(dataset+"/" + filename + ".arff")));

     /*
    //split to 70:30 learn and test set
    double percent = 70.0;
    int trainSize = (int) Math.round(data.numInstances() * percent / 100);
    int testSize = data.numInstances() - trainSize;
    Instances train = new Instances(data, 0, trainSize);
    Instances test = new Instances(data, trainSize, testSize);
    train.setClassIndex(-1);
    test.setClassIndex(-1);
*/
    System.out.println("K_means " + dataset + " " +filename);
    String[] k = {"2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26"};
    for( int i = 0; i < k.length; i++)
    {
      options = new String[2];
      options[0] = "-N";                 // max. iterations
      options[1] = k[i];
      SimpleKMeans clusterer = new SimpleKMeans();   // new instance of clusterer
      clusterer.setOptions(options);     // set the options
      AddCluster addCluster = new AddCluster();
      addCluster.setClusterer(clusterer);
      addCluster.setInputFormat(data);
      Instances kmeans_data = Filter.useFilter(data, addCluster);
      clusterer = (SimpleKMeans) addCluster.getClusterer();
      CSVSaver saver = new CSVSaver();
      saver.setInstances(kmeans_data);
      saver.setFile(new File("out/" + dataset + "/" + filename + "_" + k[i]+"_kmeans_set.csv"));
      saver.writeBatch();
    }


    
    System.out.println("EM " + dataset + " " +filename);
    String[] iterations = {"100"}; //, "20", "30", "40", "50", "60", "70", "80", "90", "100"};
    for( int i = 0; i < iterations.length; i++)
    {
      options = new String[2];
      options[0] = "-I";                 // max. iterations
      options[1] = iterations[i];
      EM clusterer = new EM();   // new instance of clusterer
      clusterer.setOptions(options);     // set the options
      
      AddCluster addCluster = new AddCluster();
      addCluster.setClusterer(clusterer);
      addCluster.setInputFormat(data);
      Instances em_data = Filter.useFilter(data, addCluster);
      CSVSaver saver = new CSVSaver();
      saver.setInstances(em_data);
      saver.setFile(new File("out/" + dataset + "/" + filename + "_" + k[i]+"_em_set.csv"));
      saver.writeBatch();
    }
    System.out.println("Finished " + dataset + " " +filename);
    
    }
    catch (Exception e)
    {
        System.out.println(e);
    }

  }
}

public class ClusteringDemo
{
  /**
   * usage:
   *   ClusteringDemo arff-file
   */
    public static void main(String[] args) throws Exception {
        String[] datasets = {"winequality-white", "magic04"};
        String[] files = {"data", "IG", "ica_1.0E-4", "pca_0.5", "rp_50"};
        for (String dataset : datasets)
        {
            System.out.println(dataset);
            for (String file : files)
            {
                ClusterThread clustering = new ClusterThread(dataset, file);
                clustering.start();
            }
        }
    }
}
