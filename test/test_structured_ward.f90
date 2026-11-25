program test_structured_ward
  use sklearn_common,only:dp,require
  use sklearn_cluster,only:agglomerative_clustering,feature_agglomeration
  use sklearn_sparse,only:csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_exceptions
  implicit none
  type(agglomerative_clustering)::model
  type(feature_agglomeration)::features
  real(dp)::x(4,1),g(4,4)
  real(dp),allocatable::z(:,:),back(:,:),large(:,:),values(:)
  integer,allocatable::ptr(:),cols(:)
  logical::flags(size(ieee_all)),modes(size(ieee_all)),after(size(ieee_all)),after_modes(size(ieee_all))
  integer::i,n,pos
  x(:,1)=[0._dp,1._dp,4._dp,5._dp];g=1
  model%connectivity=g;model%compute_full_tree='false'
  call model%fit(x)
  call require(all(model%labels_==[0,0,1,1]),'partial Ward partition')
  call require(size(model%children_,1)==2.and..not.model%compute_full_tree_,'partial Ward tree')
  call require(all(model%children_(:,1)==[0,2]).and.all(model%children_(:,2)==[1,3]),'zero-based estimator children')
  call require(model%n_clusters_==2.and.model%n_features_in_==1.and.model%n_leaves_==4,'Ward fitted metadata')
  model%compute_distances=.false.;call model%fit(x)
  call require(size(model%distances_)==0,'refit clears omitted distances')
  model%n_clusters=0;model%distance_threshold=1._dp;model%compute_full_tree='auto'
  call model%fit(x)
  call require(model%n_clusters_==4.and.size(model%distances_)==3,'equal-height threshold retains separate leaves')
  model%distance_threshold=1.01_dp;call model%fit(x)
  call require(model%n_clusters_==2.and.all(model%labels_==[0,0,1,1]),'threshold cuts below final merge')
  model%distance_threshold=10._dp;call model%fit(x)
  call require(model%n_clusters_==1.and.all(model%labels_==0),'threshold merges all leaves')
  deallocate(model%connectivity,model%distance_threshold);model%n_clusters=2
  model%connectivity_mask=g/=0;call model%fit(x)
  call require(model%compute_full_tree_.and.size(model%children_,1)==3,'auto full tree for small cluster count')
  deallocate(model%connectivity_mask);g=0;model%connectivity_csr=csr_from_dense(g)
  call model%fit(x)
  call require(model%n_connected_components_==4.and.model%connectivity_completed_,'estimator completion metadata')
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  call model%fit(x)
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after).and.all(modes.eqv.after_modes),'estimator IEEE state preservation')
  features%connectivity_csr=csr_from_dense(g);features%n_clusters=0;features%distance_threshold=1.01_dp
  z=features%fit_transform(transpose(x));back=features%inverse_transform(z)
  call require(features%n_clusters_==2.and.features%n_connected_components_==4,'feature Ward threshold metadata')
  call require(all(features%labels_==[0,0,1,1]),'feature Ward canonical labels')
  call require(maxval(abs(z(1,:)-[.5_dp,4.5_dp]))<1.e-14_dp,'feature Ward pooled means')
  call require(all(back(1,:)==[.5_dp,.5_dp,4.5_dp,4.5_dp]),'feature Ward inverse')
  ! Above the 2% heuristic boundary, auto builds a partial sparse forest.
  n=32768;allocate(large(n,1),ptr(n+1),cols(2*n-2),values(2*n-2))
  large(:,1)=[(real(i,dp),i=1,n)];pos=1;ptr(1)=1;values=1
  do i=1,n
    if(i>1)then
      cols(pos)=i-1;pos=pos+1
    end if
    if(i<n)then
      cols(pos)=i+1;pos=pos+1
    end if
    ptr(i+1)=pos
  end do
  model%connectivity_csr=csr_from_arrays(n,n,ptr,cols,values);model%n_clusters=n
  call model%fit(large)
  call require(.not.model%compute_full_tree_.and.size(model%children_,1)==0,'large auto partial tree')
  call require(all(model%labels_==[(i-1,i=1,n)]),'large sparse estimator identity partition')
  print *,'Structured Ward estimators passed; avoided an 8 GiB dense distance/connectivity matrix'
end program
