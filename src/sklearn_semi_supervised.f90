! Graph label diffusion, following sklearn/semi_supervised/_label_propagation.py.
module sklearn_semi_supervised
  use sklearn_common
  use sklearn_metrics, only: rbf_kernel,pairwise_distances,accuracy_score
  implicit none
  private
  public :: label_propagation,label_spreading
  type :: label_diffusion
    character(16) :: kernel='rbf'
    real(dp) :: gamma=20,tol=1.e-3_dp
    integer :: n_neighbors=7,n_iter_=0
    logical :: converged_=.false.
    integer, allocatable :: classes_(:),transduction_(:)
    real(dp), allocatable :: x_(:,:),label_distributions_(:,:)
  contains
    procedure :: predict_proba=>diffusion_proba
    procedure :: predict=>diffusion_predict
    procedure :: score=>diffusion_score
  end type
  type, extends(label_diffusion) :: label_propagation
    integer :: max_iter=1000
  contains
    procedure :: fit=>propagation_fit
  end type
  type, extends(label_diffusion) :: label_spreading
    integer :: max_iter=30
    real(dp) :: alpha=0.2_dp
  contains
    procedure :: fit=>spreading_fit
  end type
contains
  function affinity(self,x,y) result(w)
    class(label_diffusion), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: w(:,:),d(:,:)
    integer, allocatable :: order(:)
    integer :: i,k
    if(self%kernel=='rbf') then
      w=rbf_kernel(x,y,self%gamma)
    else
      k=self%n_neighbors
      call require(k>0.and.k<=size(y,1),'invalid semi-supervised n_neighbors')
      d=pairwise_distances(x,y); allocate(w(size(x,1),size(y,1))); w=0
      do i=1,size(x,1)
        order=argsort(d(i,:)); w(i,order(:k))=1
      end do
    end if
  end function
  subroutine diffusion_fit(self,x,y,spreading,alpha,max_iter)
    class(label_diffusion), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),alpha
    integer, intent(in) :: y(:),max_iter
    logical, intent(in) :: spreading
    real(dp), allocatable :: graph(:,:),fixed(:,:),previous(:,:),degree(:),normalizer(:)
    integer :: i,j,n,c,it
    call check_xy(x,size(y)); n=size(y)
    call require(any(y/=-1),'semi-supervised fit requires at least one labeled sample')
    call require(self%kernel=='rbf'.or.self%kernel=='knn','unknown semi-supervised kernel')
    call require(self%gamma>=0.and.self%tol>0.and.max_iter>0,'invalid label diffusion controls')
    self%classes_=unique_labels(pack(y,y/=-1)); self%x_=x; c=size(self%classes_)
    graph=affinity(self,x,x)
    if(spreading) then
      do i=1,n
        graph(i,i)=0
      end do
      degree=sqrt(sum(graph,dim=1)); where(degree==0) degree=1
      graph=graph/(spread(degree,1,n)*spread(degree,2,n))
    else
      ! sklearn normalizes rows using the affinity's column sums. This matters
      ! for the directed k-neighbor graph; RBF affinities are symmetric.
      degree=sum(graph,dim=1); where(degree==0) degree=1
      graph=graph/spread(degree,2,n)
    end if
    if(allocated(self%label_distributions_)) deallocate(self%label_distributions_)
    allocate(self%label_distributions_(n,c),fixed(n,c),previous(n,c))
    self%label_distributions_=0
    do i=1,n
      if(y(i)==-1) cycle
      self%label_distributions_(i,label_index(self%classes_,y(i)))=1
    end do
    fixed=self%label_distributions_; if(spreading) fixed=(1-alpha)*fixed
    previous=0; self%converged_=.false.
    do it=0,max_iter-1
      if(sum(abs(self%label_distributions_-previous))<self%tol) then
        self%converged_=.true.; exit
      end if
      previous=self%label_distributions_; self%label_distributions_=matmul(graph,previous)
      if(spreading) then
        self%label_distributions_=alpha*self%label_distributions_+fixed
      else
        normalizer=sum(self%label_distributions_,dim=2); where(normalizer==0) normalizer=1
        self%label_distributions_=self%label_distributions_/spread(normalizer,2,c)
        do i=1,n
          if(y(i)/=-1) self%label_distributions_(i,:)=fixed(i,:)
        end do
      end if
    end do
    self%n_iter_=it
    normalizer=sum(self%label_distributions_,dim=2); where(normalizer==0) normalizer=1
    self%label_distributions_=self%label_distributions_/spread(normalizer,2,c)
    self%transduction_=spread(0,1,n)
    do i=1,n
      self%transduction_(i)=self%classes_(maxloc(self%label_distributions_(i,:),dim=1))
    end do
  end subroutine
  subroutine propagation_fit(self,x,y)
    class(label_propagation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call diffusion_fit(self,x,y,.false.,0._dp,self%max_iter)
  end subroutine
  subroutine spreading_fit(self,x,y)
    class(label_spreading), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call require(self%alpha>0.and.self%alpha<1,'LabelSpreading alpha must lie in (0,1)')
    call diffusion_fit(self,x,y,.true.,self%alpha,self%max_iter)
  end subroutine
  function diffusion_proba(self,x) result(prob)
    class(label_diffusion), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),w(:,:)
    real(dp) :: total
    integer :: i
    call require(allocated(self%label_distributions_),'label diffusion model is not fitted')
    w=affinity(self,x,self%x_); prob=matmul(w,self%label_distributions_)
    do i=1,size(x,1)
      total=sum(prob(i,:))
      ! A disconnected/unreachable query has no label evidence. Reject rather
      ! than divide by zero and silently return NaN probabilities.
      call require(total>0,'query has no label evidence; adjust graph connectivity or gamma')
      prob(i,:)=prob(i,:)/total
    end do
  end function
  function diffusion_predict(self,x) result(y)
    class(label_diffusion), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(prob(i,:),dim=1))
    end do
  end function
  real(dp) function diffusion_score(self,x,y) result(v)
    class(label_diffusion), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
end module
