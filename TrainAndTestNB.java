import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class TrainAndTestNB {
    private int dimension;
    private Set<String> vocabulary;
    private double prior[];
    private Map<String, double[]> condprob;
    private Set<String> stopwords;
    private File trainPath;
    private File testPath;
    private int numAllDocs;

    private void getStopwords() throws IOException {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File("english.txt")));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String contentLine;
        while ((contentLine = br.readLine()) != null) {
            stopwords.add(contentLine);
        }
        if (br != null) {
            br.close();
        }
        return;
    }

    public TrainAndTestNB(String trainPath, String testPath, int dimension) throws IOException {
        this.vocabulary = new HashSet<>();
        this.trainPath = new File(trainPath);
        this.testPath = new File(testPath);
        this.condprob = new HashMap<>();
        this.stopwords = new HashSet<>();
        this.dimension = dimension;
        this.prior = new double[dimension];
        getStopwords();
        numAllDocs = countDocs(new File(trainPath));
    }

    private void extractVocabulary(File dir) throws IOException {
        if (dir.getName().startsWith("."))
            return;

        if (dir.isFile()) {
            BufferedReader br = new BufferedReader(new FileReader(dir));
            String contentLine;
            while ((contentLine = br.readLine()) != null) {
                if (contentLine.startsWith("Lines:")) {
                    break;
                }
            }
            while ((contentLine = br.readLine()) != null) {
                for (String str : contentLine.split("[^a-zA-Z']+")) {
                    if (!stopwords.contains(str)) {
                        vocabulary.add(str);
                    }
                }
            }
            return;
        }

        File[] list = dir.listFiles();

        if (list == null)
            return;

        for (File file : list) {
            extractVocabulary(file);
        }
    }

    public int countDocs(File dir) {
        int cnt = 0;
        if (dir.isFile())
            return 1;
        for (File file : dir.listFiles()) {

            if (file.getName().startsWith("."))
                continue;

            if (file.isDirectory()) {
                cnt += countDocs(file);
            } else
                cnt++;
        }
        return cnt;
    }

    public void trainingNB() throws IOException {
        int i = 0;
        extractVocabulary(trainPath);

        File[] flist = trainPath.listFiles();

        if (flist == null) return;

        for (File file : flist) {
            if (file.getName().startsWith("."))
                continue;
            int numDocClass = countDocs(file);
            prior[i] = (double) numDocClass / (double) numAllDocs;

            HashMap<String, Integer> Tct = new HashMap<>();

            for (File f : file.listFiles()) {
                BufferedReader br = new BufferedReader(new FileReader(f));
                String contentLine;
                while ((contentLine = br.readLine()) != null) {
                    if (contentLine.startsWith("Lines:")) {
                        break;
                    }
                }
                while ((contentLine = br.readLine()) != null) {
                    for (String str : contentLine.split("[^a-zA-Z']+")) {
                        if (!stopwords.contains(str)) {
                            if (Tct.containsKey(str))
                                Tct.put(str, Tct.get(str) + 1);
                            else
                                Tct.put(str, 1);
                        }
                    }
                }
            }

            int sum = 0;
            for (Integer v : Tct.values()) {
                sum += (v + 1);
                //condprob.put(str, condprob.get(str).add)
            }
            for (String str : vocabulary) {
                double d = ((Tct.getOrDefault(str, 0) + 1) / (double) sum);
                if (!condprob.containsKey(str)) {
                    double[] list = new double[this.dimension];
                    list[i] = d;
                    condprob.put(str, list);
                } else {
                    condprob.get(str)[i] = d;
                }
            }

            i++;

            if (i >= dimension)
                break;
        }

    }

    private Set<String> extractTokenInDoc(File file) throws IOException {
        Set<String> set = new HashSet<>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        String contentLine;
        while ((contentLine = br.readLine()) != null) {
            if (contentLine.startsWith("Lines:")) {
                break;
            }
        }
        while ((contentLine = br.readLine()) != null) {
            for (String str : contentLine.split("[^a-zA-Z']+")) {
                if (vocabulary.contains(str)) {
                    set.add(str);
                }
            }
        }
        return set;
    }

    public int applyNB(File file) throws IOException {
        Set<String> set = extractTokenInDoc(file);
        double score[] = new double[dimension];
        double max = Double.NEGATIVE_INFINITY;
        int index = -1;
        for (int i = 0; i < dimension; i++) {
            score[i] = Math.log(prior[i]);
            for (String str : set) {
                score[i] += Math.log(condprob.get(str)[i]);
            }
            if (score[i] > max) {
                max = score[i];
                index = i;
            }
        }
        return index;
    }
}
