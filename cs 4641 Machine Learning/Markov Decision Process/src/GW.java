import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

public class GW{

	public static void main(String[] args) throws IOException {
		boolean[] sizes = {false, true};
		for (boolean size: sizes) {
			double[] prob = {0.99, 0.9, 0.8, 0.6, 0.4, 0.2};
			double[] wallpercent = {0.05, 0.1, 0.2};
			double[] reward = {-0.1, 1, 3, 5, 10, 100};
			double[] discount = {0.99, 0.95, 0.9, 0.8, 0.6};
			double[] qInit = {0.3, 0.5, 1, 5, 30};
			double[] epsilon = {0.05, 0.1, 0.3, 0.5, 0.8};

			OutputStream outs = System.out;
			PrintStream dos = new PrintStream(outs);
			System.setOut(dos);
            String sizeStr = size ? "large/" : "small/";
            System.out.println(sizeStr);

			String head = "prob";

			System.setOut(dos);
			System.out.println(head);
			new File("./output/" + sizeStr+ head).mkdirs();
			File file = new File("./output/" + sizeStr+ head + "/out_GW"); //Your file
			FileOutputStream fos = new FileOutputStream(file);
			PrintStream ps = new PrintStream(fos);
			System.setOut(ps);
			
			for (int i = 0; i < prob.length; i++) {
				System.out.println("----------------------" + head + "_" + prob[i] + "-----------------------");
				MyMap map = new MyMap(prob[i], size);
				MyGridWorld gw = new MyGridWorld(map, 5, -0.1, 0.99);
				System.out.println("-------------Value Iteration-------------");
				gw.valueIteration("./output/" + sizeStr+ head + "/GridWorld_" + prob[i]);
				System.out.println("-------------Policy Iteration-------------");
				gw.policyIteration("./output/" + sizeStr+ head + "/GridWorld_" + prob[i]);
			}

			head = "reward";
			System.setOut(dos);
			System.out.println(head);
			new File("./output/" + sizeStr+ head).mkdirs();
			file = new File("./output/" + sizeStr+ head + "/out_GW"); //Your file
			fos = new FileOutputStream(file);
			ps = new PrintStream(fos);
			System.setOut(ps);
			for (int i = 0; i < reward.length; i++) {
				System.out.println("----------------------" + head + "_" + reward[i] + "-----------------------");
				MyMap map = new MyMap(0.8, size);
				MyGridWorld gw = new MyGridWorld(map, reward[i], -0.1, 0.99);
				System.out.println("-------------Value Iteration-------------");
				gw.valueIteration("./output/"+ sizeStr+ head + "/GridWorld_" + reward[i]);
				System.out.println("-------------Policy Iteration-------------");
				gw.policyIteration("./output/" + sizeStr+ head +"/GridWorld_" + reward[i]);
			}

			head = "discount";
			System.setOut(dos);
			System.out.println(head);
			new File("./output/" + sizeStr+ head).mkdirs();
			file = new File("./output/" + sizeStr+ head + "/out_GW"); //Your file
			fos = new FileOutputStream(file);
			ps = new PrintStream(fos);
			System.setOut(ps);
			for (int i = 0; i < discount.length; i++) {
				System.out.println("----------------------" + head + "_" + discount[i] + "-----------------------");
				MyMap map = new MyMap(0.8, size);
				MyGridWorld gw = new MyGridWorld(map, 5, -0.1, discount[i]);
				System.out.println("-------------Value Iteration-------------");
				gw.valueIteration("./output/" + sizeStr+ head +"/GridWorld_" + discount[i]);
				System.out.println("-------------Policy Iteration-------------");
				gw.policyIteration("./output/" + sizeStr+ head + "/GridWorld_" + discount[i]);
			}



			head = "qInit";
			System.setOut(dos);
			System.out.println(head);
			new File("./output/" + sizeStr+ head).mkdirs();
			file = new File("./output/" + sizeStr+ head + "/out_GW"); //Your file
			fos = new FileOutputStream(file);
			ps = new PrintStream(fos);
			System.setOut(ps);
			for (int i = 0; i < qInit.length; i++) {
				System.out.println("----------------------" + head + "_" + qInit[i] + "-----------------------");
				MyMap map = new MyMap(0.8, size);
				MyGridWorld gw = new MyGridWorld(map, 5, -0.1, 0.99);
				System.out.println("-------------Q Learning-------------");
				gw.QLearning("./output/" + sizeStr+ head + "/GridWorld_" + qInit[i], qInit[i], 0.1, 0.1);
				System.out.println("-------------ANY KEY-------------");
			}


			head = "epsilon";
			System.setOut(dos);
			System.out.println(head);
			new File("./output/" + sizeStr+ head).mkdirs();
			file = new File("./output/" + sizeStr+ head + "/out_GW"); //Your file
			fos = new FileOutputStream(file);
			ps = new PrintStream(fos);
			System.setOut(ps);
			for (int i = 0; i < epsilon.length; i++) {
				System.out.println("----------------------" + head + "_" + epsilon[i] + "-----------------------");
				MyMap map = new MyMap(0.8, size);
				MyGridWorld gw = new MyGridWorld(map, 5, -0.1, 0.99);
				System.out.println("-------------Q Learning-------------");
				gw.QLearning("./output/" + sizeStr+ head + "/GridWorld_" + epsilon[i], 0.3, 0.1, epsilon[i]);
				System.out.println("-------------ANY KEY-------------");
			}
			System.setOut(dos);
			System.out.println("ALL DONE");

		}
	}

}
