import java.io.File;
import java.io.IOException;

public class Driver {
    public static void main(String[] args) throws IOException {
        int dimension = 5;
        if (args.length >= 3)
            dimension = Integer.parseInt(args[2]);

        TrainAndTestNB nb = new TrainAndTestNB(args[0], args[1], dimension);
        nb.trainingNB();
        int numTestDocs = nb.countDocs(new File(args[1]));
        int i = 0;
        int error = 0;
        for (File file : new File(args[1]).listFiles()) {
            if (file.getName().startsWith("."))
                continue;
            for (File f : file.listFiles()) {
                if (f.getName().startsWith("."))
                    continue;
                if (nb.applyNB(f) != i) {
                    error++;
                }
            }
            i++;
            if (i >= dimension)
                break;
        }
        System.out.println("The test accuracy is " + (1 - error / (double)numTestDocs));
        return;
    }
}
