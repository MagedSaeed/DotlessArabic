if [ ! -d kenlm ]; then
  wget https://kheafield.com/code/kenlm.tar.gz | tar -xvzf
  mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
else
  echo kenlm already exist
fi