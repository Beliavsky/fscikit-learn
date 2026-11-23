program test_tsne_bh
  use sklearn, only: tsne,dp,csr_matrix,csr_from_dense,pairwise_distances
  use sklearn_common, only: require
  use sklearn_tsne_bh, only: tsne_bh_probabilities,tsne_bh_objective
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  type(tsne) :: model
  type(csr_matrix) :: distances,p,graph
  real(dp) :: a(4,4),position(4,2),error,x(12,3),initial(12,2)
  real(dp), allocatable :: gradient(:,:),values(:,:),z(:,:),again(:,:),dense(:,:)
  integer, allocatable :: cells(:,:)
  integer :: i,d
  a=1
  do i=1,4
    a(i,i)=0
  end do
  distances=csr_from_dense(a,.true.)
  call tsne_bh_probabilities(distances,3._dp,p)
  call require(size(p%data)==12.and.all(abs(p%data-1._dp/12)<1.e-14_dp),'uniform sparse joint probabilities')
  position=0
  call tsne_bh_objective(position,p,1,.5_dp,error,gradient,tree_indices=cells,tree_values=values)
  call require(size(cells,1)==1.and.cells(1,12)==4,'identical positions share a spatial leaf')
  call require(all(gradient==0).and.ieee_is_finite(error),'collapsed Barnes-Hut force')
  position(:,1)=[-.2_dp,.8_dp,.3_dp,-.5_dp];position(:,2)=[.1_dp,.7_dp,-.3_dp,.6_dp]
  call tsne_bh_objective(position,p,1,0._dp,error,gradient,skip=2,compute_error=.false.)
  call require(all(gradient(:2,:)==0).and.ieee_is_nan(error),'skipped points and omitted objective')
  do i=1,12
    x(i,:)=[real(i,dp),sin(real(i,dp)),cos(real(i,dp))]
    initial(i,:)=[.01_dp*i,.02_dp*mod(i,3)]
  end do
  call require(model%method=='barnes_hut','source-compatible default method')
  model%perplexity=2;model%early_exaggeration=1;model%learning_rate=.5_dp
  model%max_iter=300;model%min_grad_norm=0
  z=model%fit_transform(x,initial);again=model%fit_transform(x,initial)
  call require(all(z==again).and.all(ieee_is_finite(z)),'deterministic Barnes-Hut fit')
  dense=pairwise_distances(x,x);graph=csr_from_dense(dense,.true.);model%metric='precomputed'
  again=model%fit_transform(graph,initial)
  call require(all(z==again),'feature and CSR distance fits agree')
  model%metric='euclidean';model%init='random'
  do d=1,3
    model%n_components=d;z=model%fit_transform(x)
    call require(all(shape(z)==[12,d]).and.all(ieee_is_finite(z)),'one through three spatial dimensions')
  end do
  model%n_components=2;model%init='pca';model%min_grad_norm=1.e10_dp
  z=model%fit_transform(x)
  call require(model%n_iter_==99,'Barnes-Hut checkpoint stopping and restart')
  print *, 'Barnes-Hut t-SNE: PASS'
end program
