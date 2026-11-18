! Dense cross-decomposition, adapted from sklearn/cross_decomposition/_pls.py.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_cross_decomposition
  use sklearn_common
  use sklearn_metrics, only: r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: pls_regression,pls_canonical,cca,pls_svd
  type :: projection_data
    integer :: n_components=2
    logical :: scale=.true.
    real(dp), allocatable :: x_mean_(:),y_mean_(:),x_std_(:),y_std_(:)
    real(dp), allocatable :: x_weights_(:,:),y_weights_(:,:),x_rotations_(:,:),y_rotations_(:,:)
  contains
    procedure :: transform=>project_x
    procedure :: transform_y=>project_y
  end type
  type, extends(projection_data) :: pls_model
    integer :: max_iter=500,n_components_fitted_=0
    real(dp) :: tol=1.e-6_dp
    character(8) :: algorithm='nipals'
    logical :: converged_=.false.
    integer, allocatable :: n_iter_(:)
    real(dp), allocatable :: x_loadings_(:,:),y_loadings_(:,:),x_scores_(:,:),y_scores_(:,:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>pls_fit
    procedure :: fit_transform=>pls_fit_transform
    procedure :: inverse_transform=>restore_x
    procedure :: inverse_transform_y=>restore_y
    procedure :: predict=>pls_predict
    procedure :: score=>pls_score
  end type
  type, extends(pls_model) :: pls_regression
  end type
  type, extends(pls_model) :: pls_canonical
  end type
  type, extends(pls_model) :: cca
  end type
  type, extends(projection_data) :: pls_svd
  contains
    procedure :: fit=>svd_fit
    procedure :: fit_transform=>svd_fit_transform
  end type
contains
  subroutine preprocess(self,x,y,xc,yc)
    class(projection_data), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable, intent(out) :: xc(:,:),yc(:,:)
    integer :: n
    call check_x(x); call check_xy(y,size(x,1)); n=size(x,1)
    call require(n>=2,'cross-decomposition requires at least two samples')
    self%x_mean_=sum(x,dim=1)/n; self%y_mean_=sum(y,dim=1)/n
    xc=x-spread(self%x_mean_,1,n); yc=y-spread(self%y_mean_,1,n)
    self%x_std_=spread(1._dp,1,size(x,2)); self%y_std_=spread(1._dp,1,size(y,2))
    if(self%scale) then
      self%x_std_=sqrt(sum(xc**2,dim=1)/(n-1)); self%y_std_=sqrt(sum(yc**2,dim=1)/(n-1))
      where(self%x_std_==0) self%x_std_=1
      where(self%y_std_==0) self%y_std_=1
    end if
    xc=xc/spread(self%x_std_,1,n); yc=yc/spread(self%y_std_,1,n)
  end subroutine
  function pinverse(a,old_cutoff) result(inv)
    real(dp), intent(in) :: a(:,:)
    logical, intent(in), optional :: old_cutoff
    real(dp) :: inv(size(a,2),size(a,1)),s(size(a,2)),u(size(a,1),size(a,2))
    real(dp) :: v(size(a,2),size(a,2)),cutoff
    integer :: j
    call dense_svd(a,s,u,v); cutoff=maxval(s)*epsilon(1._dp)*maxval(shape(a))
    if(present(old_cutoff)) then
      if(old_cutoff) cutoff=maxval(s)*1.e6_dp*epsilon(1._dp)
    end if
    inv=0
    do j=1,minval(shape(a))
      if(s(j)>cutoff) inv=inv+spread(v(:,j)/s(j),2,size(a,1))*spread(u(:,j),1,size(a,2))
    end do
  end function
  subroutine orient(a,b)
    real(dp), intent(inout) :: a(:),b(:)
    integer :: j
    j=maxloc(abs(a),dim=1)
    if(a(j)<0) then
      a=-a; b=-b
    end if
  end subroutine
  subroutine leading_svd(x,y,a,b)
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), intent(out) :: a(:),b(:)
    real(dp) :: cross(size(x,2),size(y,2)),s(size(y,2))
    real(dp) :: u(size(x,2),size(y,2)),v(size(y,2),size(y,2))
    cross=matmul(transpose(x),y); call dense_svd(cross,s,u,v)
    call require(s(1)>tiny(1._dp),'cross-decomposition zero cross-covariance')
    a=u(:,1); b=v(:,1)
  end subroutine
  subroutine pls_fit(self,x,y)
    class(pls_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: xc(:,:),yc(:,:),xp(:,:),yp(:,:),a(:),b(:),old(:),t(:),u(:),loadx(:),loady(:)
    integer, allocatable :: iterations(:)
    real(dp) :: eps,denom
    integer :: n,p,q,k,c,j,it,upper
    logical :: canonical,mode_b,done
    canonical=.false.; mode_b=.false.
    select type(self)
    type is(pls_canonical); canonical=.true.
    type is(cca); canonical=.true.; mode_b=.true.
    end select
    call preprocess(self,x,y,xc,yc); n=size(x,1); p=size(x,2); q=size(y,2); k=self%n_components
    upper=min(n,p)
    if(canonical) upper=min(upper,q)
    call require(k>=1.and.k<=upper,'invalid cross-decomposition n_components')
    call require(self%max_iter>0.and.ieee_is_finite(self%tol).and.self%tol>=0, &
      'invalid cross-decomposition iteration controls')
    call require(self%algorithm=='nipals'.or.(self%algorithm=='svd'.and.canonical.and..not.mode_b), &
      'unsupported cross-decomposition algorithm')
    self%x_weights_=spread(spread(0._dp,1,p),2,k); self%y_weights_=spread(spread(0._dp,1,q),2,k)
    self%x_loadings_=self%x_weights_; self%y_loadings_=self%y_weights_
    self%x_scores_=spread(spread(0._dp,1,n),2,k); self%y_scores_=self%x_scores_
    allocate(a(p),b(q),old(p),t(n),u(n),loadx(p),loady(q),iterations(k))
    self%n_components_fitted_=0; self%converged_=.true.; eps=epsilon(1._dp); iterations=0
    do c=1,k
      if(self%algorithm=='svd') then
        call leading_svd(xc,yc,a,b)
      else
        do j=1,q
          if(all(abs(yc(:,j))<10*eps)) yc(:,j)=0
        end do
        j=1
        do while(j<=q)
          if(any(abs(yc(:,j))>eps)) exit
          j=j+1
        end do
        if(j>q) exit
        u=yc(:,j); old=100; done=.false.
        if(mode_b) then
          xp=pinverse(xc,.true.); yp=pinverse(yc,.true.)
        end if
        do it=1,self%max_iter
          if(mode_b) then
            a=matmul(xp,u)
          else
            a=matmul(transpose(xc),u)/sum(u*u)
          end if
          a=a/(sqrt(sum(a*a))+eps); t=matmul(xc,a)
          call require(sum(t*t)>tiny(1._dp),'cross-decomposition degenerate predictor scores')
          if(mode_b) then
            b=matmul(yp,t)
          else
            b=matmul(transpose(yc),t)/sum(t*t)
          end if
          if(canonical) b=b/(sqrt(sum(b*b))+eps)
          call require(sum(b*b)>tiny(1._dp),'cross-decomposition degenerate response weights')
          u=matmul(yc,b)/(sum(b*b)+eps)
          if(sum((a-old)**2)<self%tol.or.q==1) then
            done=.true.; exit
          end if
          old=a
        end do
        iterations(c)=min(it,self%max_iter); self%converged_=self%converged_.and.done
      end if
      call orient(a,b); t=matmul(xc,a); denom=sum(b*b)
      if(canonical) denom=1
      call require(denom>tiny(1._dp),'cross-decomposition degenerate response weights')
      u=matmul(yc,b)/denom
      call require(sum(t*t)>tiny(1._dp),'cross-decomposition degenerate predictor scores')
      loadx=matmul(transpose(xc),t)/sum(t*t); xc=xc-spread(t,2,p)*spread(loadx,1,n)
      if(canonical) then
        call require(sum(u*u)>tiny(1._dp),'cross-decomposition degenerate response scores')
        loady=matmul(transpose(yc),u)/sum(u*u); yc=yc-spread(u,2,q)*spread(loady,1,n)
      else
        loady=matmul(transpose(yc),t)/sum(t*t); yc=yc-spread(t,2,q)*spread(loady,1,n)
      end if
      self%x_weights_(:,c)=a; self%y_weights_(:,c)=b
      self%x_loadings_(:,c)=loadx; self%y_loadings_(:,c)=loady
      self%x_scores_(:,c)=t; self%y_scores_(:,c)=u; self%n_components_fitted_=c
    end do
    if(self%algorithm=='nipals') then
      self%n_iter_=iterations(:self%n_components_fitted_)
    else
      self%n_iter_=iterations(:0)
    end if
    self%x_rotations_=matmul(self%x_weights_,pinverse(matmul(transpose(self%x_loadings_),self%x_weights_)))
    self%y_rotations_=matmul(self%y_weights_,pinverse(matmul(transpose(self%y_loadings_),self%y_weights_)))
    self%coef_=transpose(matmul(self%x_rotations_,transpose(self%y_loadings_)))* &
      spread(self%y_std_,2,p)/spread(self%x_std_,1,q)
    self%intercept_=self%y_mean_
  end subroutine
  subroutine svd_fit(self,x,y)
    class(pls_svd), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: xc(:,:),yc(:,:),cross(:,:),s(:),u(:,:),v(:,:)
    integer :: k,p,q,j
    call preprocess(self,x,y,xc,yc); k=self%n_components; p=size(x,2); q=size(y,2)
    call require(k>=1.and.k<=min(size(x,1),p,q),'invalid cross-decomposition n_components')
    allocate(s(q),u(p,q),v(q,q)); cross=matmul(transpose(xc),yc); call dense_svd(cross,s,u,v)
    call require(s(k)>epsilon(1._dp)*max(p,q)*s(1),'PLSSVD requires retained full cross-covariance rank')
    do j=1,k
      call orient(u(:,j),v(:,j))
    end do
    self%x_weights_=u(:,:k); self%y_weights_=v(:,:k)
    self%x_rotations_=self%x_weights_; self%y_rotations_=self%y_weights_
  end subroutine
  function project_x(self,x) result(z)
    class(projection_data), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%x_rotations_),'cross-decomposition is not fitted')
    call check_xy(transpose(x),size(self%x_rotations_,1))
    z=matmul((x-spread(self%x_mean_,1,size(x,1)))/spread(self%x_std_,1,size(x,1)),self%x_rotations_)
  end function
  function project_y(self,y) result(z)
    class(projection_data), intent(in) :: self
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%y_rotations_),'cross-decomposition is not fitted')
    call check_xy(transpose(y),size(self%y_rotations_,1))
    z=matmul((y-spread(self%y_mean_,1,size(y,1)))/spread(self%y_std_,1,size(y,1)),self%y_rotations_)
  end function
  function restore_x(self,z) result(x)
    class(pls_model), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%x_loadings_),'cross-decomposition is not fitted')
    call check_xy(transpose(z),size(self%x_loadings_,2))
    x=matmul(z,transpose(self%x_loadings_))*spread(self%x_std_,1,size(z,1))+spread(self%x_mean_,1,size(z,1))
  end function
  function restore_y(self,z) result(y)
    class(pls_model), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: y(:,:)
    call require(allocated(self%y_loadings_),'cross-decomposition is not fitted')
    call check_xy(transpose(z),size(self%y_loadings_,2))
    y=matmul(z,transpose(self%y_loadings_))*spread(self%y_std_,1,size(z,1))+spread(self%y_mean_,1,size(z,1))
  end function
  function pls_predict(self,x) result(y)
    class(pls_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call require(allocated(self%coef_),'cross-decomposition is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    y=matmul(x-spread(self%x_mean_,1,size(x,1)),transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  real(dp) function pls_score(self,x,y) result(value)
    class(pls_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: prediction(:,:)
    integer :: j
    prediction=self%predict(x); call check_xy(y,size(x,1))
    call require(size(y,2)==size(prediction,2),'cross-decomposition target width mismatch')
    value=0
    do j=1,size(y,2)
      value=value+r2_score(y(:,j),prediction(:,j))
    end do
    value=value/size(y,2)
  end function
  function pls_fit_transform(self,x,y) result(z)
    class(pls_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,y); z=self%transform(x)
  end function
  function svd_fit_transform(self,x,y) result(z)
    class(pls_svd), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,y); z=self%transform(x)
  end function
end module
