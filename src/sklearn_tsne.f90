! Exact/Barnes-Hut t-SNE and trustworthiness: sklearn 1.6.1 manifold/_t_sne.py.
! BSD-3-Clause adaptation; see THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_tsne
  use sklearn_common, only: dp,require,check_x,argsort
  use sklearn_metrics, only: pairwise_distances
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_tsne_bh, only: tsne_bh_probabilities
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_tsne_core, only: tsne_joint_probabilities,tsne_exact_objective,tsne_gradient_descent,tsne_sum32
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
    character(16) :: method='barnes_hut',init='pca'
    real(dp), allocatable :: embedding_(:,:)
    real(dp) :: angle=.5_dp
  contains
    procedure, private :: fit_dense=>tsne_fit,fit_sparse=>tsne_fit_sparse
    generic :: fit=>fit_dense,fit_sparse
    procedure, private :: map_dense=>tsne_fit_transform,map_sparse=>tsne_map_sparse
    generic :: fit_transform=>map_dense,map_sparse
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
  subroutine parameters(self,n)
    class(tsne), intent(in) :: self
    integer, intent(in) :: n
    integer :: k
    k=self%n_components
    call require(n>=2,'t-SNE requires at least two samples')
    call require(k>0,'t-SNE n_components must be positive')
    call require(self%method=='exact'.or.self%method=='barnes_hut','invalid t-SNE method')
    if(self%method=='barnes_hut') call require(k<=3,'Barnes-Hut t-SNE requires one to three components')
    call require(self%max_iter>=250,'t-SNE max_iter must be at least 250')
    call require(self%n_iter_without_progress>0,'t-SNE progress patience must be positive')
    call require(ieee_is_finite(self%early_exaggeration),'t-SNE early exaggeration must be finite')
    call require(self%early_exaggeration>=1,'t-SNE early exaggeration must be at least one')
    call require(ieee_is_finite(self%learning_rate),'t-SNE learning rate must be finite')
    call require(self%learning_rate==-1.or.self%learning_rate>0,'invalid t-SNE learning rate')
    call require(ieee_is_finite(self%min_grad_norm),'t-SNE gradient threshold must be finite')
    call require(self%min_grad_norm>=0,'t-SNE gradient threshold must be nonnegative')
    call require(ieee_is_finite(self%perplexity),'t-SNE perplexity must be finite')
    call require(self%perplexity>0.and.self%perplexity<n,'t-SNE requires 0 < perplexity < n_samples')
    call require(real(self%perplexity,real32)>0,'t-SNE perplexity is below float32 range')
    call require(ieee_is_finite(self%angle),'t-SNE angle must be finite')
    call require(self%angle>=0.and.self%angle<=1,'t-SNE angle must be in [0,1]')
  end subroutine
  subroutine tsne_fit(self,x,init)
    class(tsne), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init(:,:)
    real(dp), allocatable :: distances(:,:),probabilities(:),position(:,:)
    type(csr_matrix) :: graph,p
    logical :: single
    integer :: n
    call check_x(x);n=size(x,1);call parameters(self,n)
    if(self%method=='exact') then
      if(self%metric=='euclidean') then
        distances=pairwise_distances(x,x,'sqeuclidean')
      else
        distances=input_distances(x,self%metric,self%p)
        call require(maxval(distances)<=sqrt(real(huge(1._real32),dp)),'t-SNE distances exceed float32 range')
        distances=distances**2
      end if
      call tsne_joint_probabilities(distances,self%perplexity,probabilities)
    else
      distances=input_distances(x,self%metric,self%p)
      graph=nearest_graph(self,csr_from_dense(distances,.true.))
      call tsne_bh_probabilities(graph,self%perplexity,p);allocate(probabilities(0))
    end if
    call initialize(self,n,position,single,init,x)
    call optimize(self,position,probabilities,single,p)
    self%n_features_in_=size(x,2)
  end subroutine
  subroutine tsne_fit_sparse(self,x,init)
    class(tsne), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in), optional :: init(:,:)
    type(csr_matrix) :: graph,p
    real(dp), allocatable :: position(:,:),empty(:)
    logical :: single
    call x%validate();call parameters(self,x%n_rows)
    call require(self%method=='barnes_hut','CSR t-SNE requires method=barnes_hut')
    call require(self%metric=='precomputed','CSR t-SNE requires metric=precomputed')
    call require(x%n_rows==x%n_cols,'manifold precomputed distances must be square')
    graph=nearest_graph(self,x);call tsne_bh_probabilities(graph,self%perplexity,p)
    call initialize(self,x%n_rows,position,single,init)
    allocate(empty(0));call optimize(self,position,empty,single,p);self%n_features_in_=x%n_cols
  end subroutine
  function nearest_graph(self,input) result(graph)
    class(tsne), intent(in) :: self
    type(csr_matrix), intent(in) :: input
    type(csr_matrix) :: graph
    integer, allocatable :: order(:)
    integer :: n,k,i,j,t,lo,hi,remove,used
    integer(int64) :: capacity
    n=input%n_rows;call input%validate()
    call require(input%n_rows==input%n_cols,'manifold precomputed distances must be square')
    call require(all(ieee_is_finite(input%data)),'Barnes-Hut distances must be finite')
    call require(all(input%data>=0),'manifold precomputed distances must be nonnegative')
    k=n-1
    if(self%perplexity<real(n-2,dp)/3) k=int(3*self%perplexity+1)
    capacity=int(n,int64)*k
    call require(capacity<huge(1),'Barnes-Hut graph exceeds integer capacity')
    graph%n_rows=n;graph%n_cols=n;allocate(graph%indptr(n+1),graph%indices(int(capacity)),graph%data(int(capacity)))
    used=0;graph%indptr(1)=1
    do i=1,n
      lo=input%indptr(i);hi=input%indptr(i+1)-1;order=argsort(input%data(lo:hi))
      call require(size(order)>=k+1,'not enough stored neighbors for Barnes-Hut t-SNE')
      remove=0
      do j=1,k+1
        if(input%indices(lo+order(j)-1)==i) then
          call require(remove==0,'duplicate selected self entries in Barnes-Hut neighbors');remove=j
        end if
      end do
      if(remove==0) remove=1
      do j=1,k+1
        if(j==remove) cycle
        t=lo+order(j)-1
        call require(input%data(t)<=sqrt(real(huge(1._real32),dp)),'t-SNE distances exceed float32 range')
        used=used+1;graph%indices(used)=input%indices(t);graph%data(used)=input%data(t)**2
      end do
      graph%indptr(i+1)=used+1
    end do
  end function
  subroutine initialize(self,n,position,single,init,x)
    class(tsne), intent(in) :: self
    integer, intent(in) :: n
    real(dp), allocatable, intent(out) :: position(:,:)
    logical, intent(out) :: single
    real(dp), optional, intent(in) :: init(:,:),x(:,:)
    type(legacy_random_state) :: rng
    real(dp), allocatable :: centered(:,:),s(:),u(:,:),vt(:,:)
    real(real32), allocatable :: initial32(:,:)
    real(real32) :: mean32,scale32
    integer :: k,i,j,index
    k=self%n_components;call rng%seed(self%random_state)
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
        call require(present(x).and.self%metric/='precomputed','t-SNE PCA initialization requires feature input')
        call require(k<=min(n,size(x,2)),'invalid t-SNE PCA component count')
        centered=x-spread(sum(x,dim=1)/n,1,n)
        call randomized_svd(centered,k,rng,s,u,vt)
        do j=1,k
          index=maxloc(abs(vt(j,:)),dim=1)
          if(vt(j,index)<0) u(:,j)=-u(:,j)
          initial32(:,j)=real(u(:,j)*s(j),real32)
        end do
        mean32=tsne_sum32(initial32(:,1))/real(n,real32)
        scale32=sqrt(tsne_sum32((initial32(:,1)-mean32)**2)/real(n,real32))
        call require(scale32>0,'t-SNE PCA initialization has zero scale')
        initial32=initial32/scale32*1.e-4_real32
      case default
        call require(.false.,'unknown t-SNE initialization')
      end select
      position=real(initial32,dp)
    end if
    call require(maxval(abs(position))<sqrt(huge(1._dp)/k)/4,'t-SNE initial embedding exceeds working range')
  end subroutine
  subroutine optimize(self,position,probabilities,single,p)
    class(tsne), intent(inout) :: self
    real(dp), intent(inout) :: position(:,:),probabilities(:)
    logical, intent(in) :: single
    type(csr_matrix), intent(inout) :: p
    real(dp) :: error
    integer :: degrees,iteration
    logical :: auto_rate
    auto_rate=self%learning_rate==-1;self%learning_rate_=self%learning_rate
    if(auto_rate) self%learning_rate_=max(real(size(position,1),dp)/self%early_exaggeration/4,50._dp)
    degrees=max(self%n_components-1,1)
    probabilities=probabilities*self%early_exaggeration
    if(self%method=='barnes_hut') p%data=p%data*self%early_exaggeration
    call phase(0,250,250,.5_dp)
    probabilities=probabilities/self%early_exaggeration
    if(self%method=='barnes_hut') p%data=p%data/self%early_exaggeration
    if(iteration<250.or.self%max_iter>250) call phase(iteration+1,self%max_iter,self%n_iter_without_progress,.8_dp)
    self%embedding_=position;self%kl_divergence_=error;self%n_iter_=iteration
  contains
    subroutine phase(first,last,patience,momentum)
      integer, intent(in) :: first,last,patience
      real(dp), intent(in) :: momentum
      if(self%method=='barnes_hut') then
        call tsne_gradient_descent(position,probabilities,degrees,first,last,patience,momentum,self%learning_rate_, &
          self%min_grad_norm,single,auto_rate,error,iteration,bh_probabilities=p,angle=self%angle)
      else
        call tsne_gradient_descent(position,probabilities,degrees,first,last,patience,momentum,self%learning_rate_, &
          self%min_grad_norm,single,auto_rate,error,iteration)
      end if
    end subroutine
  end subroutine
  function tsne_fit_transform(self,x,init) result(embedding)
    class(tsne), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init(:,:)
    real(dp), allocatable :: embedding(:,:)
    call self%fit(x,init);embedding=self%embedding_
  end function
  function tsne_map_sparse(self,x,init) result(embedding)
    class(tsne), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in), optional :: init(:,:)
    real(dp), allocatable :: embedding(:,:)
    call self%fit(x,init);embedding=self%embedding_
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
