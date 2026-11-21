! Dense exact t-SNE and trustworthiness: sklearn 1.6.1 manifold/_t_sne.py.
! BSD-3-Clause adaptation; see THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_tsne
  use sklearn_common, only: dp,require,check_x,argsort
  use sklearn_metrics, only: pairwise_distances
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_tsne_core, only: tsne_joint_probabilities,tsne_exact_objective,tsne_gradient_descent
  use, intrinsic :: iso_fortran_env, only: real32,int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: tsne,trustworthiness
  type :: tsne
    integer :: n_components=2,max_iter=1000,n_iter_without_progress=300,random_state=0
    integer :: n_iter_=0,n_features_in_=0
    real(dp) :: perplexity=30,early_exaggeration=12,learning_rate=-1,min_grad_norm=1.e-7_dp,p=2
    real(dp) :: kl_divergence_=0,learning_rate_=0
    character(32) :: metric='euclidean'
    character(16) :: method='exact',init='pca'
    real(dp), allocatable :: embedding_(:,:)
  contains
    procedure :: fit=>tsne_fit
    procedure :: fit_transform=>tsne_fit_transform
  end type
contains
  function input_distances(x,metric,p) result(distances)
    real(dp), intent(in) :: x(:,:),p
    character(*), intent(in) :: metric
    real(dp), allocatable :: distances(:,:)
    call check_x(x)
    if(metric=='precomputed') then
      call require(size(x,1)==size(x,2),'manifold precomputed distances must be square')
      call require(all(x>=0),'manifold precomputed distances must be nonnegative')
      distances=x
    else
      distances=pairwise_distances(x,x,metric,p)
    end if
  end function
  subroutine tsne_fit(self,x,init)
    class(tsne), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init(:,:)
    type(legacy_random_state) :: rng
    real(dp), allocatable :: distances(:,:),probabilities(:),position(:,:),centered(:,:),s(:),u(:,:),vt(:,:)
    real(real32), allocatable :: initial32(:,:)
    real(real32) :: mean32,scale32
    real(dp) :: error
    integer :: n,k,i,j,degrees,iteration,index
    logical :: single,auto_rate
    call check_x(x); n=size(x,1); k=self%n_components
    call require(n>=2,'t-SNE requires at least two samples')
    call require(k>0,'t-SNE n_components must be positive')
    call require(self%method=='exact','only exact t-SNE is implemented')
    call require(self%max_iter>=250,'t-SNE max_iter must be at least 250')
    call require(self%n_iter_without_progress>0,'t-SNE progress patience must be positive')
    call require(ieee_is_finite(self%early_exaggeration),'t-SNE early exaggeration must be finite')
    call require(self%early_exaggeration>=1,'t-SNE early exaggeration must be at least one')
    call require(ieee_is_finite(self%learning_rate),'t-SNE learning rate must be finite')
    call require(self%learning_rate==-1.or.self%learning_rate>0,'invalid t-SNE learning rate')
    call require(ieee_is_finite(self%min_grad_norm),'t-SNE gradient threshold must be finite')
    call require(self%min_grad_norm>=0,'t-SNE gradient threshold must be nonnegative')
    call rng%seed(self%random_state)
    if(self%metric=='euclidean') then
      distances=pairwise_distances(x,x,'sqeuclidean')
    else
      distances=input_distances(x,self%metric,self%p)
      call require(maxval(distances)<=sqrt(real(huge(1._real32),dp)),'t-SNE distances exceed float32 range')
      distances=distances**2
    end if
    call tsne_joint_probabilities(distances,self%perplexity,probabilities)
    deallocate(distances)
    allocate(position(n,k)); single=.not.present(init)
    if(present(init)) then
      call check_x(init)
      call require(size(init,1)==n.and.size(init,2)==k,'t-SNE initial embedding shape mismatch')
      position=init
    else
      allocate(initial32(n,k))
      select case(trim(self%init))
      case('random')
        do i=1,n
          do j=1,k
            initial32(i,j)=real(rng%normal(),real32)*1.e-4_real32
          end do
        end do
      case('pca')
        call require(self%metric/='precomputed','t-SNE PCA initialization requires feature input')
        call require(k<=min(n,size(x,2)),'invalid t-SNE PCA component count')
        centered=x-spread(sum(x,dim=1)/n,1,n)
        call randomized_svd(centered,k,rng,s,u,vt)
        do j=1,k
          index=maxloc(abs(vt(j,:)),dim=1)
          if(vt(j,index)<0) u(:,j)=-u(:,j)
          initial32(:,j)=real(u(:,j)*s(j),real32)
        end do
        mean32=sum(initial32(:,1))/real(n,real32)
        scale32=sqrt(sum((initial32(:,1)-mean32)**2)/real(n,real32))
        call require(scale32>0,'t-SNE PCA initialization has zero scale')
        initial32=initial32/scale32*1.e-4_real32
      case default
        call require(.false.,'unknown t-SNE initialization')
      end select
      position=real(initial32,dp)
    end if
    call require(maxval(abs(position))<sqrt(huge(1._dp)/k)/4,'t-SNE initial embedding exceeds working range')
    auto_rate=self%learning_rate==-1; self%learning_rate_=self%learning_rate
    if(auto_rate) self%learning_rate_=max(real(n,dp)/self%early_exaggeration/4,50._dp)
    degrees=max(k-1,1); probabilities=probabilities*self%early_exaggeration
    call tsne_gradient_descent(position,probabilities,degrees,0,250,250,.5_dp,self%learning_rate_, &
      self%min_grad_norm,single,auto_rate,error,iteration)
    probabilities=probabilities/self%early_exaggeration
    if(iteration<250.or.self%max_iter>250) then
      call tsne_gradient_descent(position,probabilities,degrees,iteration+1,self%max_iter, &
        self%n_iter_without_progress,.8_dp,self%learning_rate_,self%min_grad_norm,single,auto_rate,error,iteration)
    end if
    self%embedding_=position; self%kl_divergence_=error; self%n_iter_=iteration; self%n_features_in_=size(x,2)
  end subroutine
  function tsne_fit_transform(self,x,init) result(embedding)
    class(tsne), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init(:,:)
    real(dp), allocatable :: embedding(:,:)
    call self%fit(x,init); embedding=self%embedding_
  end function
  real(dp) function trustworthiness(x,x_embedded,n_neighbors,metric,p) result(value)
    real(dp), intent(in) :: x(:,:),x_embedded(:,:)
    integer, intent(in), optional :: n_neighbors
    character(*), intent(in), optional :: metric
    real(dp), intent(in), optional :: p
    character(:), allocatable :: selected
    real(dp), allocatable :: distances(:,:),embedded_distances(:,:)
    integer, allocatable :: order(:),neighbors(:),ranks(:)
    real(dp) :: exponent
    integer(int64) :: penalty
    integer :: n,k,i,j
    call check_x(x); call check_x(x_embedded); n=size(x,1); k=5; selected='euclidean'; exponent=2
    if(present(n_neighbors)) k=n_neighbors
    if(present(metric)) selected=metric
    if(present(p)) exponent=p
    call require(size(x_embedded,1)==n,'trustworthiness sample count mismatch')
    call require(k>=1.and.real(k,dp)<real(n,dp)/2,'trustworthiness requires 1 <= n_neighbors < n_samples/2')
    distances=input_distances(x,selected,exponent); embedded_distances=pairwise_distances(x_embedded,x_embedded)
    allocate(ranks(n)); penalty=0
    do i=1,n
      distances(i,i)=huge(1._dp); embedded_distances(i,i)=huge(1._dp)
      order=argsort(distances(i,:)); neighbors=argsort(embedded_distances(i,:))
      do j=1,n
        ranks(order(j))=j
      end do
      do j=1,k
        penalty=penalty+max(0,ranks(neighbors(j))-k)
      end do
    end do
    value=1-real(penalty,dp)*(2/(real(n,dp)*k*(2*real(n,dp)-3*real(k,dp)-1)))
  end function
end module
