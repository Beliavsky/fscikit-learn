program test_linkage_tree
  use sklearn_common,only:dp,require
  use sklearn_hierarchy,only:linkage_tree,hierarchy_result
  use sklearn_cluster,only:agglomerative_clustering,feature_agglomeration
  use sklearn_sparse,only:csr_from_dense,csr_from_arrays,csr_matrix
  use,intrinsic::ieee_exceptions
  implicit none
  type(hierarchy_result)::tree,other
  type(agglomerative_clustering)::model
  type(feature_agglomeration)::features
  type(csr_matrix)::sparse
  real(dp)::x(4),g(4,4),d(4,4),boundary(2,3)
  real(dp),allocatable::large(:),values(:),z(:,:)
  integer,allocatable::ptr(:),columns(:)
  logical::flags(size(ieee_all)),modes(size(ieee_all)),after(size(ieee_all)),after_modes(size(ieee_all))
  integer::i,j,n
  x=[0._dp,2._dp,5._dp,9._dp];g=0
  do i=1,3
    g(i,i+1)=1
  end do
  tree=linkage_tree(x,return_distance=.true.)
  call require(all(tree%children(:,1)==[1,3,5]).and.all(tree%children(:,2)==[2,4,6]),'complete chain golden')
  call require(all(tree%distances==[2._dp,4._dp,9._dp]),'complete linkage maximum distances')
  other=linkage_tree(reshape(x,[4,1]),return_distance=.true.,linkage='average')
  call require(all(other%children(:,1)==[1,3,4]).and.all(other%children(:,2)==[2,5,6]),'average chain golden')
  call require(maxval(abs(other%distances-[2._dp,4._dp,20._dp/3]))<1.e-14_dp,'average weighted recurrence')
  tree=linkage_tree(x,g,return_distance=.true.,linkage='average')
  call require(all(tree%distances==[2._dp,3._dp,4._dp]),'missing neighbors copy distances without reweighting')
  call require(all(tree%parents==[5,5,6,7,6,7,7]),'structured complete/average parent tree')
  other=linkage_tree(x,g/=0,return_distance=.true.)
  call require(all(other%children==tree%children).and.all(other%distances==tree%distances),'logical vector wrapper')
  sparse=csr_from_dense(g)
  other=linkage_tree(x,sparse,n_clusters=2,return_distance=.true.)
  call require(size(other%children,1)==2.and.all(other%parents==[5,5,6,4,6,6]),'partial CSR vector parent forest')
  other=linkage_tree(reshape(x,[4,1]),g,return_distance=.true.)
  call require(all(other%children==tree%children),'real matrix wrapper')
  other=linkage_tree(reshape(x,[4,1]),g/=0,return_distance=.true.)
  call require(all(other%children==tree%children),'logical matrix wrapper')
  other=linkage_tree(reshape(x,[4,1]),sparse,n_clusters=2)
  call require(size(other%distances)==0.and.size(other%children,1)==2,'CSR matrix distance omission')
  do i=1,4
    do j=1,4
      d(i,j)=abs(x(i)-x(j))
      if(j<=i)d(i,j)=100
    end do
  end do
  other=linkage_tree(d,affinity='precomputed',return_distance=.true.)
  call require(all(other%distances==[2._dp,4._dp,9._dp]),'unstructured precomputed consumes only upper triangle')
  model%linkage='complete';model%connectivity_csr=sparse;model%compute_full_tree='false'
  call model%fit(reshape(x,[4,1]))
  call require(size(model%children_,1)==2.and.all(model%labels_==[0,0,0,1]),'fitted partial complete linkage')
  features%linkage='average';features%connectivity_csr=sparse;features%n_clusters=0;features%distance_threshold=3
  z=features%fit_transform(reshape(x,[1,4]))
  call require(features%n_clusters_==3.and.all(features%labels_==[0,0,1,2]),'feature average threshold cut')
  call require(all(z(1,:)==[1._dp,5._dp,9._dp]),'average-linkage feature pooling')
  ! Source paired Euclidean uses NumPy einsum, not a scalar left fold. A one-
  ! ulp difference here changes the cluster count exactly at this threshold.
  boundary(1,:)=[-.59440035219873522_dp,-1.4805365125650163_dp,-.88813385373365239_dp]
  boundary(2,:)=[1.9510125601262995_dp,2.0769805290537531_dp,.069381135132778701_dp]
  deallocate(model%connectivity_csr);model%connectivity=spread([1._dp,1._dp],2,2)
  model%compute_full_tree='true';model%n_clusters=0;model%distance_threshold=4.4779336026277639_dp
  do i=1,2
    if(i==1)then
      model%linkage='complete'
    else
      model%linkage='average'
    end if
    call model%fit(boundary)
    call require(model%n_clusters_==2.and.model%distances_(1)==model%distance_threshold, &
      'NumPy paired-norm rounding preserves exact threshold equality')
  end do
  call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  other=linkage_tree(x,sparse,return_distance=.true.,linkage='average')
  call model%fit(boundary)
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after).and.all(modes.eqv.after_modes),'complete/average IEEE state restoration')
  n=8192;allocate(large(n),ptr(n+1),columns(n-1),values(n-1))
  large=[(real(i,dp),i=1,n)];ptr=[(i,i=1,n),n];columns=[(i,i=2,n)];values=1
  sparse=csr_from_arrays(n,n,ptr,columns,values)
  do i=1,2
    if(i==1)then
      other=linkage_tree(large,sparse,n_clusters=n-1,return_distance=.true.)
    else
      other=linkage_tree(large,sparse,n_clusters=n-1,return_distance=.true.,linkage='average')
    end if
    call require(size(other%children,1)==1.and.all(other%children(1,:)==[8191,8192]),'large cost-only heap golden')
    call require(other%distances(1)==1.and.other%n_connected_components==1,'large sparse partial linkage')
  end do
  print *,'Complete/average trees passed; avoided a 512 MiB dense distance/connectivity matrix'
end program
