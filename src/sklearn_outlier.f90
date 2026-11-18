! Local Outlier Factor, matching the dense sklearn.neighbors estimator subset.
module sklearn_outlier
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances
  implicit none
  private
  public :: local_outlier_factor
  type :: local_outlier_factor
    integer :: n_neighbors=20,n_neighbors_=0
    real(dp) :: contamination=-1,p=2,offset_=-1.5_dp
    character(16) :: metric='minkowski'
    logical :: novelty=.false.
    real(dp), allocatable :: x_(:,:),negative_outlier_factor_(:),lrd_(:),k_distance_(:)
  contains
    procedure :: fit=>lof_fit
    procedure :: fit_predict=>lof_fit_predict
    procedure :: score_samples=>lof_samples
    procedure :: decision_function=>lof_decision
    procedure :: predict=>lof_predict
  end type
contains
  subroutine lof_fit(self,x)
    class(local_outlier_factor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: d(:,:),distances(:,:)
    integer, allocatable :: indices(:,:),order(:)
    integer :: i,j,n,k,lo,hi
    real(dp) :: position
    call check_x(x); n=size(x,1); call require(n>1,'LOF requires at least two samples')
    call require(self%n_neighbors>0,'LOF n_neighbors must be positive')
    call require(self%contamination==-1.or.(self%contamination>0.and.self%contamination<=0.5_dp), &
      'LOF contamination must be -1 (auto) or in (0,0.5]')
    self%x_=x; k=min(self%n_neighbors,n-1); self%n_neighbors_=k
    d=pairwise_distances(x,x,self%metric,self%p); allocate(indices(n,k),distances(n,k))
    do i=1,n
      d(i,i)=huge(1._dp); order=argsort(d(i,:))
      indices(i,:)=order(:k); distances(i,:)=d(i,order(:k))
    end do
    self%k_distance_=distances(:,k); self%lrd_=spread(0._dp,1,n)
    self%negative_outlier_factor_=self%lrd_
    do i=1,n
      self%lrd_(i)=1/(sum(max(distances(i,:),self%k_distance_(indices(i,:))))/k+1.e-10_dp)
    end do
    do i=1,n
      self%negative_outlier_factor_(i)=-sum(self%lrd_(indices(i,:)))/(k*self%lrd_(i))
    end do
    self%offset_=-1.5_dp
    if(self%contamination/=-1) then
      order=argsort(self%negative_outlier_factor_); position=1+(n-1)*self%contamination
      lo=floor(position); hi=ceiling(position)
      self%offset_=self%negative_outlier_factor_(order(lo))+(position-lo)* &
        (self%negative_outlier_factor_(order(hi))-self%negative_outlier_factor_(order(lo)))
    end if
  end subroutine
  function lof_fit_predict(self,x) result(labels)
    class(local_outlier_factor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call require(.not.self%novelty,'LOF fit_predict requires novelty=false')
    call self%fit(x); labels=merge(1,-1,self%negative_outlier_factor_>=self%offset_)
  end function
  function lof_samples(self,x) result(scores)
    class(local_outlier_factor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1)),lrd
    real(dp), allocatable :: d(:,:)
    integer, allocatable :: order(:)
    integer :: i,k
    call require(allocated(self%x_),'local_outlier_factor is not fitted')
    call require(self%novelty,'LOF query scoring requires novelty=true')
    d=pairwise_distances(x,self%x_,self%metric,self%p); k=self%n_neighbors_
    do i=1,size(x,1)
      order=argsort(d(i,:)); order=order(:k)
      lrd=1/(sum(max(d(i,order),self%k_distance_(order)))/k+1.e-10_dp)
      scores(i)=-sum(self%lrd_(order))/(k*lrd)
    end do
  end function
  function lof_decision(self,x) result(scores)
    class(local_outlier_factor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1))
    scores=self%score_samples(x)-self%offset_
  end function
  function lof_predict(self,x) result(labels)
    class(local_outlier_factor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    labels=merge(1,-1,self%decision_function(x)>=0)
  end function
end module
