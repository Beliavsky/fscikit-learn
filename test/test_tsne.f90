program test_tsne
  use sklearn, only: tsne,trustworthiness,dp
  use sklearn_tsne_core, only: tsne_joint_probabilities,tsne_exact_objective,tsne_gradient_descent
  use sklearn_common, only: require
  use, intrinsic :: iso_fortran_env, only: real32
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(tsne) :: model
  real(dp) :: distances(4,4),position(4,2),plus(4,2),minus(4,2),error,ep,em
  real(dp) :: x(8,3),initial(8,2),original(8,3),old_init(8,2),delta
  real(dp), allocatable :: p(:),g(:,:),work(:,:),z(:,:),again(:,:)
  integer :: i,j,iteration
  model%method='exact'
  distances=1
  do i=1,4
    distances(i,i)=0
  end do
  call tsne_joint_probabilities(distances,3._dp,p)
  call require(size(p)==6.and.all(abs(p-1._dp/12)<1.e-14_dp),'uniform pair probabilities')
  call require(abs(sum(p)-.5_dp)<1.e-14_dp,'condensed probabilities sum to half')
  position=0; call tsne_exact_objective(position,p,1,error,g)
  call require(abs(error)<1.e-14_dp.and.maxval(abs(g))<1.e-14_dp,'uniform zero embedding objective')
  position(:,1)=[-.2_dp,.8_dp,.3_dp,-.5_dp]; position(:,2)=[.1_dp,.7_dp,-.3_dp,.6_dp]
  call tsne_exact_objective(position,p,1,error,g); delta=1.e-6_dp
  do i=1,4
    do j=1,2
      plus=position; minus=position; plus(i,j)=plus(i,j)+delta; minus(i,j)=minus(i,j)-delta
      call tsne_exact_objective(plus,p,1,ep,work); call tsne_exact_objective(minus,p,1,em,work)
      call require(abs(g(i,j)-(ep-em)/(2*delta))<1.e-8_dp,'finite-difference exact t-SNE gradient')
    end do
  end do
  position=0
  call tsne_gradient_descent(position,p,1,0,1,300,.5_dp,50._dp,0._dp,.false.,.false.,error,iteration)
  call require(iteration==0.and.all(position==0),'zero-gradient optimizer fixed point')
  position(:,1)=[1._dp,-2._dp,3._dp,-4._dp]*1.e-200_dp
  position(:,2)=position(:,1)*.5_dp
  call tsne_gradient_descent(position,[.1_dp,.1_dp,.1_dp,.1_dp,.05_dp,.05_dp],1,0,100,300, &
    .5_dp,.001_dp,0._dp,.false.,.false.,error,iteration)
  call require(iteration==99,'scaled gradient norm avoids premature underflow stopping')
  do i=1,8
    x(i,:)=[real(i,dp),sin(real(i,dp)),cos(real(i,dp))]
    initial(i,:)=[.01_dp*i,.02_dp*mod(i,3)]
  end do
  original=x; old_init=initial; model%perplexity=2; model%early_exaggeration=1
  model%learning_rate=.5_dp; model%max_iter=300; model%min_grad_norm=0
  z=model%fit_transform(x,initial); again=model%fit_transform(x,initial)
  call require(all(shape(z)==[8,2]).and.all(ieee_is_finite(z)),'finite fitted embedding')
  call require(all(z==again).and.all(z==model%embedding_),'deterministic refit and stored embedding')
  call require(all(x==original).and.all(initial==old_init),'fit preserves feature and initialization arrays')
  call require(model%n_iter_==299.and.model%n_features_in_==3,'source zero-based iteration and feature counts')
  call require(model%learning_rate_==.5_dp.and.ieee_is_finite(model%kl_divergence_),'fitted optimizer diagnostics')
  model%max_iter=250; call model%fit(x,initial)
  call require(model%n_iter_==250.and.model%kl_divergence_==huge(1._dp),'source empty second-phase convention')
  model%max_iter=300; model%learning_rate=-1; model%init='random'; model%early_exaggeration=12
  z=model%fit_transform(x)
  call require(all(ieee_is_finite(z)).and.model%learning_rate_==50,'random initialization and auto learning rate')
  call require(all(z==real(real(z,real32),dp)),'random initialization follows float32 position state')
  model%init='pca'; model%perplexity=7; model%early_exaggeration=1; model%min_grad_norm=1.e10_dp
  z=model%fit_transform(x)
  call require(model%n_iter_==99,'checkpoint stopping and second-phase restart')
  call require(all(z==real(real(z,real32),dp)),'PCA initialization follows float32 position state')
  ep=trustworthiness(reshape([0._dp,1._dp,3._dp,6._dp,10._dp],[5,1]), &
    reshape([0._dp,10._dp,1._dp,3._dp,6._dp],[5,1]),1)
  call require(abs(ep-2._dp/3)<1.e-14_dp,'independent trustworthiness rank penalty')
  ep=trustworthiness(x,x,2); call require(ep==1,'identity embedding trustworthiness')
  ep=trustworthiness(x,spread([0._dp,0._dp],1,8),2)
  call require(ep>=0.and.ep<=1,'collapsed embedding stable tie contract')
  print *, 'exact t-SNE and trustworthiness: PASS'
end program
