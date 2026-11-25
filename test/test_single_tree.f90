program test_single_tree
  use sklearn_common,only:dp,require
  use sklearn_hierarchy,only:linkage_tree,hierarchy_result
  use sklearn_cluster,only:agglomerative_clustering,feature_agglomeration
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_exceptions
  implicit none
  type(hierarchy_result)::tree,other
  type(agglomerative_clustering)::model
  type(feature_agglomeration)::features
  type(csr_matrix)::sparse
  real(dp)::x(4),g(4,4),d(3,3),one(1,1)
  real(dp),allocatable::large(:),values(:),z(:,:)
  integer,allocatable::ptr(:),columns(:)
  logical::flags(size(ieee_all)),modes(size(ieee_all)),after(size(ieee_all)),after_modes(size(ieee_all))
  integer::i,n
  x=[0._dp,4._dp,5._dp,10._dp];g=0
  do i=1,3
    g(i,i+1)=1
  end do
  tree=linkage_tree(x,linkage='single',return_distance=.true.)
  call require(all(tree%children(:,1)==[2,1,6]).and.all(tree%children(:,2)==[3,5,4]),'oriented fast-MST children')
  call require(all(tree%distances==[1._dp,4._dp,5._dp]),'streamed single-linkage heights')
  other=linkage_tree(reshape(x,[4,1]),linkage='single',affinity='sqeuclidean',return_distance=.true.)
  call require(all(other%children(:,1)==[2,1,4]).and.all(other%children(:,2)==[3,5,6]),'SciPy fallback child sorting')
  call require(all(other%distances==[1._dp,16._dp,25._dp]),'squared-distance fallback heights')
  tree=linkage_tree(x,g,n_clusters=3,linkage='single',return_distance=.true.)
  call require(size(tree%children,1)==3.and.size(tree%distances)==3,'partial single retains full children/distances')
  call require(size(tree%parents)==5.and.all(tree%parents==[1,5,5,4,5]),'partial single parents retain only first merge')
  other=linkage_tree(x,g/=0,n_clusters=4,linkage='single')
  call require(size(other%children,1)==3.and.size(other%distances)==0,'logical vector wrapper full children')
  call require(all(other%parents==[1,2,3,4]),'unmerged single forest parents')
  sparse=csr_from_dense(g)
  other=linkage_tree(x,sparse,linkage='single',return_distance=.true.)
  call require(all(other%children==tree%children),'CSR vector single wrapper')
  other=linkage_tree(reshape(x,[4,1]),g,linkage='single')
  call require(all(other%children==tree%children),'real matrix single wrapper')
  other=linkage_tree(reshape(x,[4,1]),g/=0,linkage='single')
  call require(all(other%children==tree%children),'logical matrix single wrapper')
  other=linkage_tree(reshape(x,[4,1]),sparse,linkage='single')
  call require(all(other%children==tree%children),'CSR matrix single wrapper')
  d=epsilon(1._dp)
  do i=1,3
    d(i,i)=9
  end do
  other=linkage_tree(d,linkage='single',affinity='precomputed',return_distance=.true.)
  call require(all(other%distances==epsilon(1._dp)),'unstructured genuine epsilon distances remain positive')
  other=linkage_tree(d,d/=0,n_clusters=3,linkage='single',affinity='precomputed',return_distance=.true.)
  call require(all(other%distances==0),'source sparse MST maps selected genuine epsilon edges to zero')
  call require(all(other%children(:,1)==[1,4]).and.all(other%children(:,2)==[2,3]),'stable CSR zero-edge labels')
  call require(all(other%parents==[1,2,3]),'epsilon full-children/unmerged-parent contract')
  one=9
  other=linkage_tree(one,one/=0,linkage='single',affinity='precomputed',return_distance=.true.)
  call require(other%n_leaves==1.and.size(other%children,1)==0.and.all(other%parents==[1]),'source precomputed singleton')
  model%linkage='single';model%connectivity_csr=sparse;model%compute_full_tree='false';model%n_clusters=3
  call model%fit(reshape(x,[4,1]))
  call require(all(model%labels_==[0,1,1,2]).and.size(model%children_,1)==3,'fitted partial single partition')
  features%linkage='single';features%connectivity_csr=sparse;features%n_clusters=0;features%distance_threshold=4
  z=features%fit_transform(reshape(x,[1,4]))
  call require(features%n_clusters_==3.and.all(features%labels_==[0,1,1,2]),'single feature threshold equality')
  call require(all(z(1,:)==[0._dp,4.5_dp,10._dp]),'single-linkage feature pooling')
  call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  other=linkage_tree(x,sparse,n_clusters=4,linkage='single',return_distance=.true.)
  call model%fit(reshape(x,[4,1]))
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after).and.all(modes.eqv.after_modes),'single tree and estimator IEEE preservation')
  n=4096;allocate(large(n));large=[(real(i,dp),i=1,n)]
  other=linkage_tree(large,linkage='single',return_distance=.true.)
  call require(size(other%children,1)==n-1.and.all(other%distances==1),'large streamed MST without condensed distances')
  call require(all(other%children(n-1,:)==[2*n-2,n]),'large streamed oriented root')
  deallocate(large);n=16384;allocate(large(n),ptr(n+1),columns(n-1),values(n-1))
  large=[(real(i,dp),i=1,n)];ptr=[(i,i=1,n),n];columns=[(i,i=2,n)];values=1
  sparse=csr_from_arrays(n,n,ptr,columns,values)
  other=linkage_tree(large,sparse,n_clusters=n,linkage='single',return_distance=.true.)
  call require(size(other%children,1)==n-1.and.size(other%parents)==n,'large sparse full-tree/partial-parent result')
  call require(all(other%parents==[(i,i=1,n)]).and.all(other%distances==1),'large sparse roots and heights')
  model%connectivity_csr=sparse;model%n_clusters=n;model%compute_full_tree='auto'
  call model%fit(reshape(large,[n,1]))
  call require(.not.model%compute_full_tree_.and.all(model%labels_==[(i-1,i=1,n)]),'large single auto forest')
  print *,'Single trees passed; streamed 4096 samples and avoided a 2 GiB dense sparse-graph distance matrix'
end program
