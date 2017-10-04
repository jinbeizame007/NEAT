using System;
using System.Collections.Generic;

namespace NEAT
{
    public class Genome{
        private int inp_n;
        private int out_n;
        private string[] activation = { "sigmoid", "tanh", "relu", "identity" };
        public Dictionary<int,(string,List<int>)> node = new Dictionary<int,(string,List<int>)>();
        public List<(int,int,double,bool)> connection = new List<(int, int, double, bool)>();

        public Genome(int inp_n, int out_n){
            this.inp_n = inp_n;
            this.out_n = out_n;
            Random rnd = new System.Random();
            for (int i = 0; i < this.inp_n; i++)
                this.AddNode(i,this.activation[rnd.Next(this.activation.Length)]);
            for (int o = 0; o < this.out_n; o++)
                this.AddNode(this.inp_n + o, "sigmoid");
            for (int i = 0; i < this.inp_n; i++)
                for (int o = 0; o < this.out_n; o++)
                    this.AddConnection(i, this.inp_n + o, rnd.NextDouble() * 2.0 - 1.0, true);
            this.Build();
        }

        public void AddNode(int n, string activation){
            this.node.Add(n, ("sigmoid"/*activation*/, new List<int>()));
        }

        public void AddConnection(int inp_n, int out_n, double weight, bool enabled){
            int i = 0;
            foreach ((int,int,double,bool) c in this.connection){
                if (inp_n == c.Item1 && out_n == c.Item2){
                    this.connection[i] = (c.Item1, c.Item2, weight, c.Item4);
                    return;
                }
                i++;
            }
            this.connection.Add((inp_n, out_n, weight, enabled));
            this.node[out_n].Item2.Add(inp_n);
        }

        public void Build(){
            /*Console.WriteLine("=======================");
            foreach ((int, int, double, bool) c in this.connection)
            {
                Console.Write(c.Item1);
                Console.Write(",");
                Console.Write(c.Item2);
                Console.Write(" ");
            }
            Console.WriteLine("");*/
            List<int> s = new List<int>();
			for (int i = 0; i < this.inp_n; i++)
				s.Add(i);
            List<(int, int, double, bool)> connection_ = new List<(int, int, double, bool)>();
            int count = 1;
            while (count != 0){
                count = 0;
                foreach ((int, int, double, bool) c in this.connection){
                    if (connection_.Contains(c)) continue;
                    bool flag = false;
                    foreach (int n in this.node[c.Item1].Item2){
                        if (s.Contains(n) == false){
                            flag = true;
                            break;
                        }
                    }
                    if (flag) continue;
                    connection_.Add(c);
                    count++;
                }
                foreach (KeyValuePair<int,(string,List<int>)> n in this.node){
                    bool flag = true;
                    foreach (int v in n.Value.Item2){
                        if (s.Contains(v) == false) {
                            flag = false;
                            break;
                        }
                    }
					if (flag) s.Add(n.Key);
                }
            }
            this.connection = new List<(int, int, double, bool)>(connection_);
			/*foreach ((int, int, double, bool) c in connection_)
			{
				Console.Write(c.Item1);
				Console.Write(",");
				Console.Write(c.Item2);
				Console.Write(" ");
			}
			Console.WriteLine("");*/
        }

        public double[] Forward(double[] x){
            Dictionary<int, double> node = new Dictionary<int, double>();
            foreach (int k in this.node.Keys) node.Add(k, 0.0);
            for (int i = 0; i < x.Length; i++)
                node[i] = x[i];
            foreach ((int,int,double,bool) c in this.connection)
                node[c.Item2] += this.Activations(this.node[c.Item1].Item1,node[c.Item1]) * c.Item3;
            double[] n = new double[this.out_n];
            double[] node_ = new double[node.Count];
            int l = 0;
            foreach(double v in node.Values) {
                node_[l] = v;
                l++;
            }
            for (int i = 0; i < this.out_n; i++)
                n[i] = node_[this.inp_n + i];
            return n;
        }

        public void CrossOver(Genome genome){
            Random rnd = new System.Random();
            foreach ((int,int,double,bool) cy in genome.connection) {
                bool exist = false;
                int i = -1;
                foreach ((int,int,double,bool) cx in this.connection) {
                    i++;
                    if (cx.Item1 == cy.Item1 && cx.Item2 == cy.Item2) {
                        double r = rnd.NextDouble();
                        exist = true;
                        if (r < 0.5) this.connection[i] = (cx.Item1, cx.Item2, cy.Item3, cx.Item4);
                        break;
                    }
                }
                if (exist == false && this.node.ContainsKey(cy.Item1) && this.node.ContainsKey(cy.Item2))
                    this.AddConnection(cy.Item1, cy.Item2, cy.Item3, cy.Item4);
            }
        }

        public int Mutate(int node_n){
            Random rnd = new System.Random();
            double r = rnd.NextDouble();
            if (r < 0.4) {
                if (r < 0.04)
                {
                    this.MutateAddNode(node_n);
                    node_n++;
                }
                else if (r < 0.08) this.MutateDeleteNode();
                else if (r < 0.24) this.MutateDeleteConnection();
                else this.MutateAddConnection();
            }
            return node_n;
        }

        public void MutateAddNode(int n){
            //if (this.connection.Count == 0) return;
            Random rnd = new System.Random();
            int r = rnd.Next(this.connection.Count);
            (int, int, double, bool) c = this.connection[r];
            this.AddNode(n,this.activation[rnd.Next(4)]);
            this.AddConnection(c.Item1, n, rnd.NextDouble() * 2.0 - 1.0, true);
            this.AddConnection(n, c.Item2, rnd.NextDouble() * 2.0 - 1.0, true);
            this.node[c.Item2].Item2.Remove(c.Item1);
            this.node[c.Item2].Item2.Add(n);
            this.connection.RemoveAt(r);
		}

        public void MutateDeleteNode(){
            List<int> available_nodes = new List<int>();
            foreach (int key in this.node.Keys)
                if (key >= this.inp_n + this.out_n) available_nodes.Add(key);
            if (available_nodes.Count == 0) return;
            Random rand = new System.Random();
            int del_key = available_nodes[rand.Next(available_nodes.Count)];
            List<int> deletes = new List<int>();
            for (int i = 0; i < this.connection.Count; i++){
                if (del_key == this.connection[i].Item1 || del_key == this.connection[i].Item2){
                    if (this.CheckOutput(this.connection[i].Item1, this.connection[i].Item2)) return;
                    deletes.Add(i);
                }
            }
            deletes.Reverse();
            foreach (int i in deletes){
                this.node[this.connection[i].Item2].Item2.Remove(this.connection[i].Item1);
                this.connection.RemoveAt(i);
            }
            this.node.Remove(del_key);
        }

        public void MutateAddConnection(){
            int inp_n = 0;
            int out_n = 0;
            int[] keys = new int[this.node.Count];
            this.node.Keys.CopyTo(keys,0);
            Random rnd = new System.Random();
            while (this.CheckCycle(keys[inp_n], keys[out_n])){
                inp_n = this.inp_n;
                while (this.inp_n <= inp_n && inp_n < this.inp_n + this.out_n)
                    inp_n = rnd.Next(this.node.Count);
                out_n = rnd.Next(this.inp_n, this.node.Count);
            }
            this.AddConnection(keys[inp_n], keys[out_n], rnd.NextDouble() * 2.0 - 1.0,  true);
        }

        public void MutateDeleteConnection(){
            //Console.WriteLine("");
            //Console.WriteLine("-------------------");
            //Console.WriteLine(this.node);
            //Console.WriteLine(this.connection.Count);
            //foreach((int, int, double, bool) c in this.connection)
            //    Console.WriteLine(c);
            if (this.connection.Count == 0) return;
            Random rnd = new System.Random();
            int key = rnd.Next(this.connection.Count);
            for (int o = 0; o < this.out_n; o++)
                if (this.connection[key].Item2 == this.inp_n + this.out_n - 1 + o && this.node[this.inp_n + this.out_n - 1 + o].Item2.Count == 1) return;
            if (this.CheckOutput(this.connection[key].Item1, this.connection[key].Item2) == false) return;
			this.node[this.connection[key].Item2].Item2.Remove(this.connection[key].Item1);
			this.connection.RemoveAt(key);
        }

        public bool CheckCycle(int inp_n,int out_n){
            if (inp_n == out_n) return true;
            List<int> visited = new List<int>();
            visited.Add(out_n);
            while (true){
                int num_added = 0;
                foreach((int,int,double,bool) c in this.connection){
                    if (visited.Contains(c.Item1) && visited.Contains(c.Item2) == false){
                        if (c.Item2 == inp_n) return true;
                        visited.Add(c.Item2);
                        num_added++;
                    }
                }
                if (num_added == 0) return false;
            }
        }

        public bool CheckOutput(int inp_n, int out_n){
            List<int> visited = new List<int>();
            for (int i = 0; i < inp_n; i++){
                visited.Add(i);
            }
            foreach ((int,int,double,bool) c in this.connection){
                if (c.Item1 == inp_n && c.Item2 == out_n) continue;
                if (visited.Contains(c.Item1)) visited.Add(c.Item2);
            }
            for (int i = 0; i < this.out_n; i++)
                if (visited.Contains(this.inp_n + this.out_n - 1 + i) == false) return false;
            return true;
        }

        public double Activations(string func, double x){
            if (func == "sigmoid"){
                x = 1.0 / (1.0 + Math.Exp(-x));
            }else if (func == "tanh"){
                x = Math.Tanh(x);
            }else if (func == "relu"){
                x = Math.Max(0.0, x);
            }
            return x;
        }

        public Genome Clone(){
            Genome genome = new Genome(inp_n, out_n);
            genome.node = new Dictionary<int, (string, List<int>)>(this.node);
			//_foreach (KeyValuePair<int, (string, List<int>)> n in this.node)
			//	genome.node.Add(n.Key, (n.Value.Item1, new List<int>(n.Value.Item2)));
			genome.connection = new List<(int, int, double, bool)>(this.connection);                
            return genome;
        }
    }
}
