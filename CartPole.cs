using System;
using WebSocketSharp;
using MathNet.Numerics.Statistics;
using System.Collections.Generic;

namespace NEAT
{
    public class CartPole
	{
        const int inp_n = 4;
        const int out_n = 1;
        const int pop_n = 150;
        public static void Main(){
            double[] obs = new double[4];
            double[] y_ = new double[1];
            double[] fitness = new double[pop_n];
            Random rnd = new System.Random();
            int p = 0;
            int node_n = inp_n + out_n;
			Genome[] pop = new Genome[pop_n];
            for (int i = 0; i < pop_n; i++)
            {
                pop[i] = new Genome(inp_n, out_n);
                pop[i].Build();
            }
            using (var ws = new WebSocket("ws://localhost:8080"))
			{
                ws.OnMessage += (sender, e) =>
                {
                    if (e.Data == "end") {
                        p++;
                        if (p == pop_n)
                        {
                            p = 0;
                            //Console.WriteLine("-----------------");
                            Console.WriteLine(fitness.Maximum());
                            //Console.WriteLine(pop[0].node.Count);
                            //Console.WriteLine(pop[0].connection.Count);
                            double ave = fitness.Mean();
                            double v = fitness.StandardDeviation();
                            for (int i = 0; i < pop_n; i++) fitness[i] = Math.Exp((fitness[i] - ave) / v);
							double sum = 0.0;
							foreach (double f in fitness) sum += f;
                            for (int i = 0; i < pop_n; i++) fitness[i] = fitness[i] / sum;
                            Genome[] new_pop = new Genome[pop_n];
                            for (int i = 0; i < pop_n; i++) new_pop[i] = pop[i].Clone();
                            for (int i = 0; i < pop_n; i++) {
                                double rx = rnd.NextDouble();
                                double ry = rnd.NextDouble();
                                int x = -1;
                                int y = -1;
                                double s = 0.0;
                                for (int j = 0; j < pop_n; j++) {
                                    s += fitness[j];
                                    if (x == -1 && rx < s) x = j;
                                    if (y == -1 && ry < s) y = j;
                                }
                                new_pop[i] = pop[x].Clone();
                                if (x == y) continue;
                                new_pop[i].CrossOver(pop[y]);
                            }
                            for (int i = 0; i < pop_n; i++)
                            {
                                node_n = new_pop[i].Mutate(node_n);
                                pop[i] = new_pop[i].Clone();
                                pop[i].Build();
                            }
                            fitness = new double[pop_n];
						}
                        ws.Send("2");
                        return;
                    }
                    string[] obs_ = e.Data.Split(',');
                    for (int i = 0; i < inp_n; i++){
                        obs[i] = double.Parse(obs_[i]);
                    }
                    y_ = pop[p].Forward(obs);
                    if (y_[0] < 0.0) {
                        ws.Send("0");
                    }else{
                        ws.Send("1");
                    }
                    fitness[p] += 1.0;
                };
				ws.Connect();
                ws.Send("2");
                Console.ReadKey(true);
            }
		}
    }
}
