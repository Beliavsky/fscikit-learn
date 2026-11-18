module sklearn_neighbors
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances,accuracy_score,r2_score
  implicit none
  private
  public :: nearest_neighbors,kneighbors_classifier,kneighbors_regressor
  type :: nearest_neighbors
    integer :: n_neighbors=5
    character(16) :: metric='minkowski'
    real(dp) :: p=2
    real(dp), allocatable :: x_(:,:)
  contains
    procedure :: fit=>neighbors_fit
    procedure :: kneighbors=>neighbors_query
    procedure :: radius_neighbors=>radius_query
    procedure :: kneighbors_graph=>neighbors_graph
  end type
  type :: kneighbors_classifier
    integer :: n_neighbors=5
    character(16) :: weights='uniform',metric='minkowski'
    real(dp) :: p=2
    type(nearest_neighbors) :: search_
    integer, allocatable :: y_(:),classes_(:)
  contains
    procedure :: fit=>knc_fit
    procedure :: predict_proba=>knc_proba
    procedure :: predict=>knc_predict
    procedure :: score=>knc_score
  end type
  type :: kneighbors_regressor
    integer :: n_neighbors=5
    character(16) :: weights='uniform',metric='minkowski'
    real(dp) :: p=2
    type(nearest_neighbors) :: search_
    real(dp), allocatable :: y_(:)
  contains
    procedure :: fit=>knr_fit
    procedure :: predict=>knr_predict
    procedure :: score=>knr_score
  end type
contains
  subroutine neighbors_fit(self,x)
    class(nearest_neighbors), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call check_x(x); call require(self%n_neighbors>0,'n_neighbors must be positive')
    self%x_=x
  end subroutine
  subroutine neighbors_query(self,x,distances,indices,n_neighbors)
    class(nearest_neighbors), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: distances(:,:)
    integer, allocatable, intent(out) :: indices(:,:)
    integer, intent(in), optional :: n_neighbors
    real(dp), allocatable :: d(:,:)
    integer, allocatable :: order(:)
    integer :: i,k
    call require(allocated(self%x_),'nearest_neighbors is not fitted')
    k=self%n_neighbors; if(present(n_neighbors)) k=n_neighbors
    call require(k>0.and.k<=size(self%x_,1),'n_neighbors exceeds training size')
    d=pairwise_distances(x,self%x_,self%metric,self%p)
    allocate(distances(size(x,1),k),indices(size(x,1),k))
    do i=1,size(x,1)
      order=argsort(d(i,:)); indices(i,:)=order(:k); distances(i,:)=d(i,order(:k))
    end do
  end subroutine
  function radius_query(self,x,radius) result(mask)
    class(nearest_neighbors), intent(in) :: self
    real(dp), intent(in) :: x(:,:),radius
    logical, allocatable :: mask(:,:)
    call require(allocated(self%x_),'nearest_neighbors is not fitted')
    call require(radius>=0,'radius must be nonnegative')
    mask=pairwise_distances(x,self%x_,self%metric,self%p)<=radius
  end function
  function neighbors_graph(self,x,mode) result(graph)
    class(nearest_neighbors), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in), optional :: mode
    real(dp), allocatable :: graph(:,:),d(:,:)
    integer, allocatable :: idx(:,:)
    integer :: i
    logical :: distance
    distance=.false.
    if(present(mode)) then
      call require(mode=='connectivity'.or.mode=='distance','unknown graph mode')
      distance=mode=='distance'
    end if
    call self%kneighbors(x,d,idx); allocate(graph(size(x,1),size(self%x_,1))); graph=0
    do i=1,size(x,1)
      if(distance) then
        graph(i,idx(i,:))=d(i,:)
      else
        graph(i,idx(i,:))=1
      end if
    end do
  end function
  subroutine knc_fit(self,x,y)
    class(kneighbors_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call check_xy(x,size(y)); self%y_=y; self%classes_=unique_labels(y)
    self%search_%n_neighbors=self%n_neighbors; self%search_%metric=self%metric; self%search_%p=self%p
    call self%search_%fit(x)
  end subroutine
  function get_weights(d,kind) result(w)
    real(dp), intent(in) :: d(:)
    character(*), intent(in) :: kind
    real(dp) :: w(size(d))
    select case(trim(kind))
    case('uniform'); w=1
    case('distance')
      if(any(d==0)) then
        w=merge(1._dp,0._dp,d==0)
      else
        w=1/d
      end if
    case default; call require(.false.,'unknown neighbor weights')
    end select
  end function
  function knc_proba(self,x) result(prob)
    class(kneighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),d(:,:),w(:)
    integer, allocatable :: idx(:,:)
    integer :: i,j,c
    call require(allocated(self%y_),'kneighbors_classifier is not fitted')
    call self%search_%kneighbors(x,d,idx)
    allocate(prob(size(x,1),size(self%classes_))); prob=0
    do i=1,size(x,1)
      w=get_weights(d(i,:),self%weights)
      do j=1,size(idx,2)
        c=label_index(self%classes_,self%y_(idx(i,j))); prob(i,c)=prob(i,c)+w(j)
      end do
      prob(i,:)=prob(i,:)/sum(w)
    end do
  end function
  function knc_predict(self,x) result(y)
    class(kneighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(prob(i,:),dim=1))
    end do
  end function
  real(dp) function knc_score(self,x,y) result(v)
    class(kneighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  subroutine knr_fit(self,x,y)
    class(kneighbors_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1])); self%y_=y
    self%search_%n_neighbors=self%n_neighbors; self%search_%metric=self%metric; self%search_%p=self%p
    call self%search_%fit(x)
  end subroutine
  function knr_predict(self,x) result(y)
    class(kneighbors_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    real(dp), allocatable :: d(:,:),w(:)
    integer, allocatable :: idx(:,:)
    integer :: i
    call require(allocated(self%y_),'kneighbors_regressor is not fitted')
    call self%search_%kneighbors(x,d,idx)
    do i=1,size(y)
      w=get_weights(d(i,:),self%weights); y(i)=dot_product(w,self%y_(idx(i,:)))/sum(w)
    end do
  end function
  real(dp) function knr_score(self,x,y) result(v)
    class(kneighbors_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
end module
