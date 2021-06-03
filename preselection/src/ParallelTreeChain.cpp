//
//  ParallelTreeChain.cpp
//  xSVJselection
//
//  Created by Jeremi Niedziela on 02/06/2021.
//

#include "ParallelTreeChain.h"

ParallelTreeChain::ParallelTreeChain()
{
  
}

ParallelTreeChain::~ParallelTreeChain()
{
  for (size_t i = 0; i < files.size(); ++i) {
    files[i]->Close();
    files[i] = nullptr;
  }
  
  for (size_t i = 0; i < trees.size(); ++i) {
    trees[i] = nullptr;
  }
}

vector<TLeaf*> ParallelTreeChain::ParallelTreeChain::FindLeaf(string &spec)
{
  return FindLeaf(spec.c_str());
}

vector<TLeaf*> ParallelTreeChain::FindLeaf(const char* spec)
{
  vector<TLeaf*> v;
  for (size_t i = 0; i < ntrees; ++i) {
    if (!Contains(spec)) {
      cout << "WARNING:: TREE DOES NOT CONTAIN SPEC " << spec << endl;
      cout << "WARNING:: ALL POINTERS WILL BE NULL" << endl;
    }
    v.push_back(trees[i]->FindLeaf(spec));
  }
  return v;
}

void ParallelTreeChain::GetN(int entry){
  int tn = 0;
  while (entry >= 0)
    entry -= sizes[tn++];
  currentTree = tn - 1;
  currentEntry = entry + sizes[tn - 1];
}

size_t ParallelTreeChain::size() {
  return ntrees;
}

int ParallelTreeChain::GetEntry(int entry) {
  if (entry > entries)
    return -1;
  GetN(entry);
  trees[currentTree]->GetEntry(currentEntry);
  return currentTree;
}

Int_t ParallelTreeChain::GetEntries() {
  return entries;
}

vector<string> ParallelTreeChain::GetTrees(string filename, string treetype) {
  GetTreeNames(filename);
  entries = 0;
  int i = 0;
  vector<string> cleanTreenames;
  for (size_t tn = 0; tn < treenames.size(); ++tn) {
    files.push_back(new TFile(treenames[tn].c_str()));
    bool hasDelphes = files[i]->GetListOfKeys()->Contains(treetype.c_str());
    if(hasDelphes) {
      trees.push_back((TTree*)files[i]->Get(treetype.c_str()));
      sizes.push_back(trees[i]->GetEntries());
      cleanTreenames.push_back(treenames[tn]);
      trees[i]->GetEntry(0);
      entries += sizes[i];
      i++;
    }
    else {
      files.pop_back();
    }
  }
  ntrees = trees.size();
  return cleanTreenames;
}

bool ParallelTreeChain::Contains(string spec) {
  // loop through trees and make sure that the spec is contained either the leaf/branch lists of eaech tree
  for (size_t i = 0; i < trees.size(); ++i) {
    if (!(trees[i]->GetListOfBranches()->Contains(spec.c_str()) || trees[i]->GetListOfLeaves()->Contains(spec.c_str()))) {
      return false;
    }
  }
  return true;
}


void ParallelTreeChain::GetTreeNames(string filename)
{
  std::ifstream file(filename.c_str());
  string s;
  while (getline(file, s))
    if (s.size() > 0)
      treenames.push_back(s);
}
