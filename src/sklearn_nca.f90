! NeighborhoodComponentsAnalysis, sklearn 1.6.1 neighbors/_nca.py.
! BSD attribution is retained in THIRD_PARTY_NOTICES.md.
module sklearn_nca
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,dense_svd,label_index
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_product
  use lbfgsb_module, only: setulb
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: neighborhood_components_analysis,nca_objective_gradient
  abstract interface
    subroutine nca_callback(transformation,iteration)
      import dp
      real(dp), intent(in) :: transformation(:)
      integer, intent(in) :: iteration
    end subroutine
  end interface
  type :: neighborhood_components_analysis
    integer :: n_components=0,max_iter=50,random_state=0
    real(dp) :: tol=1.e-5_dp
    character(16) :: init='auto'
    logical :: warm_start=.false.,converged_=.false.
    real(dp), allocatable :: init_matrix(:,:),components_(:,:),initial_components_(:,:),gradient_(:,:)
    integer :: n_features_in_=0,n_iter_=0,n_optimizer_iter_=0,n_function_evaluations_=0
    real(dp) :: objective_=0
    character(60) :: optimization_message_='',line_search_status_=''
    real(dp) :: line_search_bracket_(2)=0,line_search_step_=0
    logical :: line_search_bracketed_=.false.
    procedure(nca_callback), pointer, nopass :: callback=>null()
  contains
    procedure :: fit=>nca_fit
    procedure :: transform=>nca_transform
    procedure :: fit_transform=>nca_fit_transform
  end type
contains
  subroutine nca_objective_gradient(x,y,components,objective,gradient)
    ! Internal numerical helper: maximize soft leave-one-out same-class probability.
    real(dp), intent(in) :: x(:,:),components(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(out) :: objective,gradient(:,:)
    real(dp), allocatable :: z(:,:),prob(:,:),weighted(:,:),sym(:,:),norms(:),success(:)
    real(dp) :: minimum,distance,bound,mx,mz
    integer :: i,j,n,k,p
    call check_xy(x,size(y));n=size(x,1);p=size(x,2);k=size(components,1)
    call require(n>=2,'NCA requires at least two samples')
    call require(k>0.and.size(components,2)==p,'NCA component shape mismatch')
    call require(all(shape(gradient)==shape(components)),'NCA gradient shape mismatch')
    call require(all(ieee_is_finite(components)),'NCA components must be finite')
    mx=maxval(abs(x))
    bound=ieee_product(ieee_product(mx,maxval(abs(components))),real(p,dp))
    call require(ieee_is_finite(bound),'NCA embedded products exceed working range')
    z=matmul(x,transpose(components));mz=maxval(abs(z))
    call require(mz<=sqrt(huge(1._dp)/(4._dp*real(k,dp))),'NCA squared distances exceed working range')
    bound=ieee_product(ieee_product(mx,mz),4._dp*real(n,dp)**2*real(p,dp))
    call require(ieee_is_finite(bound),'NCA gradient products exceed working range')
    norms=sum(z*z,dim=2)
    allocate(prob(n,n),success(n),weighted(n,n),sym(n,n));prob=0
    do i=1,n
      minimum=huge(1._dp)
      do j=1,n
        if(i==j) cycle
        distance=max(0._dp,norms(i)-2*dot_product(z(i,:),z(j,:))+norms(j))
        prob(i,j)=distance;minimum=min(minimum,distance)
      end do
      do j=1,n
        if(i/=j) prob(i,j)=exp(-(prob(i,j)-minimum))
      end do
      prob(i,:)=prob(i,:)/sum(prob(i,:))
      success(i)=sum(prob(i,:),mask=y==y(i))
      weighted(i,:)= -prob(i,:)*success(i)
      where(y==y(i)) weighted(i,:)=weighted(i,:)+prob(i,:)
    end do
    objective=sum(success);sym=weighted+transpose(weighted)
    do i=1,n
      sym(i,i)=-sum(weighted(:,i))
    end do
    gradient=2*matmul(matmul(transpose(z),sym),x)
    call require(all(ieee_is_finite(gradient)),'NCA gradient is nonfinite')
  end subroutine
  function initialize(self,x,y) result(a)
    class(neighborhood_components_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: a(:,:),centered(:,:),s(:),u(:,:),v(:,:),means(:,:),std(:),scalings(:,:)
    real(dp), allocatable :: between(:,:),bs(:),bu(:,:),bv(:,:),bar(:)
    integer, allocatable :: classes(:),counts(:)
    integer :: n,p,k,c,i,j,row,r,rank,nc
    character(16) :: method
    type(legacy_random_state) :: rng
    n=size(x,1);p=size(x,2);k=self%n_components
    call require(k>=0.and.k<=p,'NCA invalid n_components')
    if(allocated(self%init_matrix)) then
      call check_x(self%init_matrix)
      r=size(self%init_matrix,1)
      call require(size(self%init_matrix,2)==p.and.r<=p,'NCA init_matrix shape mismatch')
      call require(k==0.or.k==r,'NCA init_matrix and n_components disagree')
    else
      call require(any(self%init==['auto            ','identity        ','random          ', &
        'pca             ','lda             ']),'NCA unknown init')
    end if
    if(self%warm_start.and.allocated(self%components_)) then
      call require(size(self%components_,2)==p,'NCA warm_start feature width mismatch')
      a=self%components_;return
    end if
    if(allocated(self%init_matrix)) then
      a=self%init_matrix;return
    end if
    if(k==0) k=p
    call require(int(p,int64)*k<=(huge(k)-1180)/25,'NCA optimizer dimension exceeds integer capacity')
    classes=unique_labels(y);nc=size(classes);method=self%init
    if(method=='auto') then
      if(k<=min(p,nc-1)) then
        method='lda'
      else if(k<min(p,n)) then
        method='pca'
      else
        method='identity'
      end if
    end if
    allocate(a(k,p));a=0
    select case(method)
    case('identity')
      do i=1,k
        a(i,i)=1
      end do
    case('random')
      call rng%seed(self%random_state)
      do i=1,k
        do j=1,p
          a(i,j)=rng%normal()
        end do
      end do
    case('pca')
      call require(k<=min(n,p),'NCA PCA components exceed sample or feature count')
      centered=x-spread(sum(x,dim=1)/n,1,n)
      allocate(s(p),u(n,p),v(p,p));call dense_svd(centered,s,u,v)
      a=transpose(v(:,:k))
    case('lda')
      call require(n>nc.and.k<=min(p,nc-1),'NCA LDA needs more samples than classes and valid components')
      allocate(means(nc,p),counts(nc),centered(n,p),std(p),bar(p));means=0;counts=0
      do i=1,n
        c=label_index(classes,y(i));means(c,:)=means(c,:)+x(i,:);counts(c)=counts(c)+1
      end do
      do c=1,nc
        means(c,:)=means(c,:)/counts(c)
      end do
      row=0
      do c=1,nc
        do i=1,n
          if(y(i)/=classes(c)) cycle
          row=row+1;centered(row,:)=x(i,:)-means(c,:)
        end do
      end do
      std=sqrt(sum((centered-spread(sum(centered,dim=1)/n,1,n))**2,dim=1)/n)
      where(std==0) std=1
      centered=centered/spread(std,1,n)/sqrt(real(n-nc,dp))
      allocate(s(p),u(n,p),v(p,p));call dense_svd(centered,s,u,v)
      r=count(s>1.e-4_dp)
      call require(r>0,'NCA LDA within-class rank is zero')
      scalings=v(:,:r)/spread(std,2,r)/spread(s(:r),1,p)
      bar=matmul(real(counts,dp)/n,means)
      between=matmul((means-spread(bar,1,nc))*spread(sqrt(real(counts,dp)/max(1,nc-1)),2,p),scalings)
      allocate(bs(r),bu(nc,r),bv(r,r));call dense_svd(between,bs,bu,bv)
      rank=count(bs>1.e-4_dp*bs(1))
      call require(rank>0,'NCA LDA between-class rank is zero')
      rank=min(k,rank);a=transpose(matmul(scalings,bv(:,:rank)))
    end select
  end function
  subroutine nca_fit(self,x,y)
    class(neighborhood_components_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, parameter :: m=10
    real(dp), allocatable :: a(:,:),grad(:,:),point(:),g(:),wa(:),lower(:),upper(:)
    integer, allocatable :: iwa(:),nbd(:)
    real(dp) :: f,dsave(29),factr
    integer :: isave(44),d,p,k
    logical :: lsave(4)
    character(60) :: task,csave
    call check_xy(x,size(y));call require(size(x,1)>=2,'NCA requires at least two samples')
    call require(self%max_iter>0,'NCA max_iter must be positive')
    call require(ieee_is_finite(self%tol),'NCA tol must be finite')
    call require(self%tol>=0,'NCA tol must be nonnegative')
    call require(self%tol<=huge(1._dp)*epsilon(1._dp),'NCA tol exceeds working range')
    call require(self%random_state>=0,'NCA random_state must be nonnegative')
    call require(maxval(abs(x))<=sqrt(huge(1._dp)/(16._dp*real(size(x),dp))),'NCA input exceeds SVD working range')
    a=initialize(self,x,y);self%initial_components_=a
    p=size(x,2);k=size(a,1)
    call require(int(p,int64)*k<=(huge(p)-1180)/25,'NCA optimizer dimension exceeds integer capacity')
    d=p*k
    call require(d<=(huge(d)-11*m*m-8*m)/(2*m+5),'NCA optimizer dimension exceeds integer capacity')
    allocate(grad(k,p),g(d),wa(2*m*d+5*d+11*m*m+8*m),iwa(3*d),nbd(d),lower(d),upper(d))
    point=reshape(transpose(a),[d]);nbd=0;lower=0;upper=0;wa=0;iwa=0
    isave=0;dsave=0;lsave=.false.;csave='';task='START';factr=self%tol/epsilon(1._dp);f=0;g=0
    self%n_iter_=1;self%n_optimizer_iter_=0;self%n_function_evaluations_=0
    self%line_search_status_='';self%line_search_bracket_=0
    self%line_search_step_=0;self%line_search_bracketed_=.false.
    do
      call setulb(d,m,point,lower,upper,nbd,f,g,factr,self%tol,wa,iwa,task,-1,csave,lsave,isave,dsave)
      if(task(:2)=='FG') then
        a=transpose(reshape(point,[p,k]))
        call nca_objective_gradient(x,y,a,f,grad)
        f=-f;g=-reshape(transpose(grad),[d])
        self%n_function_evaluations_=self%n_function_evaluations_+1
      else if(task(:5)=='NEW_X') then
        self%line_search_status_=csave;self%line_search_bracket_=dsave(26:27)
        self%line_search_step_=dsave(14);self%line_search_bracketed_=isave(43)==1
        self%n_optimizer_iter_=self%n_optimizer_iter_+1
        if(associated(self%callback)) call self%callback(point,self%n_iter_)
        self%n_iter_=self%n_iter_+1
        if(self%n_optimizer_iter_>=self%max_iter) then
          task='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT'
        else if(self%n_function_evaluations_>15000) then
          task='STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT'
        end if
      else
        exit
      end if
    end do
    self%components_=transpose(reshape(point,[p,k]));self%n_features_in_=p
    self%converged_=task(:4)=='CONV';self%optimization_message_=task
    call nca_objective_gradient(x,y,self%components_,self%objective_,grad);self%gradient_=grad
  end subroutine
  function nca_transform(self,x) result(z)
    class(neighborhood_components_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'NCA is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'NCA transform feature width mismatch')
    call require(ieee_is_finite(ieee_product(ieee_product(maxval(abs(x)),maxval(abs(self%components_))), &
      real(size(x,2),dp))),'NCA transform exceeds working range')
    z=matmul(x,transpose(self%components_))
    call require(all(ieee_is_finite(z)),'NCA transform exceeds working range')
  end function
  function nca_fit_transform(self,x,y) result(z)
    class(neighborhood_components_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,y);z=self%transform(x)
  end function
end module
