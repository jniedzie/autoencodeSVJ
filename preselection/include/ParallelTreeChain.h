
#include "Helpers.hpp"

class ParallelTreeChain{
public:
  ParallelTreeChain();
  ~ParallelTreeChain();
  
  vector<TLeaf*> FindLeaf(string &spec);
  
  vector<TLeaf*> FindLeaf(const char* spec);
  
  void GetN(int entry);
  
  size_t size();
  
  int GetEntry(int entry);
  
  Int_t GetEntries();
  
  vector<string> GetTrees(string filename, string treetype);
  
  bool Contains(string spec);
  
  int currentEntry, currentTree;
  
private:
  
  void GetTreeNames(string filename);
  
  size_t ntrees;
  Int_t entries;
  vector<string> treenames;
  vector<TTree*> trees;
  vector<TFile*> files;
  vector<size_t> sizes;
}; 
