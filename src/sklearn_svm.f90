! SVC contracts: sklearn 1.6.1 svm/_base.py, _classes.py and bundled LIBSVM.
! Translated nonshrinking LIBSVM weighted SMO; retained BSD notices in LICENSES/libsvm-COPYRIGHT.
module sklearn_svm
  use sklearn_nu_classification, only: nu_svc
  use sklearn_linear_svm, only: linear_svc,linear_svr
  use sklearn_common
  use sklearn_kernel_svm, only: svr,nu_svr,one_class_svm
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel,accuracy_score
  use,intrinsic::iso_fortran_env,only:real32
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public :: nu_svc
  public :: linear_svc,linear_svr
  public :: svc,svr,nu_svr,one_class_svm
  type :: binary_svm
    integer :: first=0,second=0
    real(dp), allocatable :: dual(:)
    real(dp) :: intercept=0,kkt_violation=0
    logical :: converged=.false.
    integer :: n_iter=0
  end type
  type :: svc
    real(dp) :: c=1,gamma=-1,coef0=0,tol=1.e-4_dp
    integer :: degree=3,max_iter=10000,n_iter_=0
    character(16) :: kernel='rbf',gamma_rule='scale',decision_function_shape='ovr'
    logical :: converged_=.false.,class_weight_balanced=.false.,break_ties=.false.
    integer :: n_features_in_=0,fit_status_=1
    integer, allocatable :: classes_(:),support_(:),support_original_(:),n_support_(:),n_iter_per_pair_(:)
    real(dp),allocatable::dual_coef_(:,:),intercept_(:),coef_(:,:),class_weight_(:),kkt_violation_(:)
    real(dp), allocatable :: x_fit_(:,:),support_vectors_(:,:)
    type(binary_svm), allocatable :: models_(:)
    real(dp) :: gamma_=0
  contains
    procedure :: fit=>svc_fit
    procedure :: decision_function=>svc_decision
    procedure :: predict=>svc_predict
    procedure :: score=>svc_score
  end type
contains
  function kernel_matrix(self,x,y) result(k)
    class(svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(trim(self%kernel))
    case('linear'); k=linear_kernel(x,y)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree,self%gamma_,self%coef0)
    case('sigmoid'); k=sigmoid_kernel(x,y,self%gamma_,self%coef0)
    case default; call require(.false.,'unknown SVC kernel')
    end select
  end function
  subroutine binary_fit(model,gram,target,c,tol,max_iter)
    type(binary_svm),intent(inout)::model
    real(dp),intent(in)::gram(:,:),target(:),c(:),tol
    integer,intent(in)::max_iter
    real(dp)::alpha(size(target)),g(size(target)),q(size(target),size(target)),y(size(target)),qd(size(target))
    real(dp)::gmax,gmax2,best,diff,quad,obj,oldi,oldj,delta,total,ci,cj,di,dj,rho,ub,lb,yg,violation
    integer::n,i,j,k,t,first,last,free_count
    ! LIBSVM C-SVC second-order selection and box updates, without shrinking.
    ! Its float Q cache is distinct from the double diagonal and query kernels.
    n=size(target);y=-target;alpha=0;g=-1
    do j=1,n
      qd(j)=gram(j,j)
      do i=1,n
        q(i,j)=real(real(y(i)*y(j)*gram(i,j),real32),dp)
      end do
    end do
    model%n_iter=0;model%converged=.false.
    do while(model%n_iter<max_iter)
      gmax=-huge(1._dp);gmax2=-huge(1._dp);first=0;last=0;best=huge(1._dp)
      do t=1,n
        if(y(t)>0)then
          if(alpha(t)<c(t).and.-g(t)>=gmax)then
            gmax=-g(t);first=t
          end if
        else
          if(alpha(t)>0.and.g(t)>=gmax)then
            gmax=g(t);first=t
          end if
        end if
      end do
      if(first==0)then
        model%converged=.true.;exit
      end if
      i=first
      do j=1,n
        if(y(j)>0)then
          if(alpha(j)<=0)cycle
          diff=gmax+g(j);gmax2=max(gmax2,g(j));quad=qd(i)+qd(j)-2*y(i)*q(j,i)
        else
          if(alpha(j)>=c(j))cycle
          diff=gmax-g(j);gmax2=max(gmax2,-g(j));quad=qd(i)+qd(j)+2*y(i)*q(j,i)
        end if
        if(diff>0)then
          if(quad<=0)quad=1.e-12_dp
          obj=-(diff*diff)/quad
          if(obj<=best)then
            best=obj;last=j
          end if
        end if
      end do
      if(gmax+gmax2<tol.or.last==0)then
        model%converged=.true.;exit
      end if
      j=last;model%n_iter=model%n_iter+1;ci=c(i);cj=c(j);oldi=alpha(i);oldj=alpha(j)
      if(y(i)/=y(j))then
        quad=qd(i)+qd(j)+2*q(j,i);if(quad<=0)quad=1.e-12_dp
        delta=(-g(i)-g(j))/quad;diff=alpha(i)-alpha(j);alpha(i)=alpha(i)+delta;alpha(j)=alpha(j)+delta
        if(diff>0)then
          if(alpha(j)<0)then
            alpha(j)=0;alpha(i)=diff
          end if
        else
          if(alpha(i)<0)then
            alpha(i)=0;alpha(j)=-diff
          end if
        end if
        if(diff>ci-cj)then
          if(alpha(i)>ci)then
            alpha(i)=ci;alpha(j)=ci-diff
          end if
        else
          if(alpha(j)>cj)then
            alpha(j)=cj;alpha(i)=cj+diff
          end if
        end if
      else
        quad=qd(i)+qd(j)-2*q(j,i);if(quad<=0)quad=1.e-12_dp
        delta=(g(i)-g(j))/quad;total=alpha(i)+alpha(j);alpha(i)=alpha(i)-delta;alpha(j)=alpha(j)+delta
        if(total>ci)then
          if(alpha(i)>ci)then
            alpha(i)=ci;alpha(j)=total-ci
          end if
        else
          if(alpha(j)<0)then
            alpha(j)=0;alpha(i)=total
          end if
        end if
        if(total>cj)then
          if(alpha(j)>cj)then
            alpha(j)=cj;alpha(i)=total-cj
          end if
        else
          if(alpha(i)<0)then
            alpha(i)=0;alpha(j)=total
          end if
        end if
      end if
      di=alpha(i)-oldi;dj=alpha(j)-oldj
      do k=1,n
        g(k)=g(k)+q(k,i)*di+q(k,j)*dj
      end do
    end do
    ub=huge(1._dp);lb=-huge(1._dp);total=0;free_count=0
    do i=1,n
      yg=y(i)*g(i)
      if(alpha(i)>=c(i))then
        if(y(i)<0)then
          ub=min(ub,yg)
        else
          lb=max(lb,yg)
        end if
      else if(alpha(i)<=0)then
        if(y(i)>0)then
          ub=min(ub,yg)
        else
          lb=max(lb,yg)
        end if
      else
        free_count=free_count+1;total=total+yg
      end if
    end do
    rho=ub/2+lb/2;if(free_count>0)rho=total/free_count
    model%dual=alpha*target;model%intercept=rho
    g=matmul(gram,model%dual)+rho-target;violation=0
    do i=1,n
      if(alpha(i)<c(i)-1.e-10_dp)violation=max(violation,-target(i)*g(i))
      if(alpha(i)>1.e-10_dp)violation=max(violation,target(i)*g(i))
    end do
    model%kkt_violation=violation
  end subroutine
  subroutine shape_controls(self)
    class(svc),intent(in)::self
    call require(self%decision_function_shape=='ovo'.or.self%decision_function_shape=='ovr','invalid SVC decision shape')
  end subroutine
  subroutine svc_fit(self,x,y,sample_weight,class_weight)
    class(svc),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),allocatable::gram(:,:),target(:),weights(:),capacity(:)
    integer,allocatable::ids(:),encoded(:),supports(:),positions(:)
    logical::support(size(y))
    integer::a,b,k,m,n,i,j,t
    real(dp)::variance,mean
    call check_xy(x,size(y));call shape_controls(self)
    call require(all(ieee_is_finite([self%c,self%tol,self%gamma,self%coef0])),'nonfinite SVC controls')
    call require(self%c>0.and.self%tol>0.and.self%max_iter>0,'invalid SVC controls')
    call require(self%gamma>=0.or.self%gamma==-1,'invalid SVC gamma')
    call require(self%gamma_rule=='scale'.or.self%gamma_rule=='auto','invalid SVC gamma rule')
    call require(self%degree>=0,'invalid SVC degree')
    self%classes_=unique_labels(y);k=size(self%classes_);n=size(y)
    call require(k>=2,'SVC requires at least two classes')
    weights=spread(1._dp,1,n)
    if(present(sample_weight))then
      call require(size(sample_weight)==n,'SVC sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite SVC sample weights')
      call require(all(sample_weight>=0),'negative SVC sample weights');weights=sample_weight
    end if
    encoded=[(label_index(self%classes_,y(i)),i=1,n)];self%class_weight_=spread(1._dp,1,k)
    if(present(class_weight))then
      call require(.not.self%class_weight_balanced,'choose balanced or explicit SVC class weights')
      call require(size(class_weight)==k,'SVC class weight count mismatch')
      call require(all(ieee_is_finite(class_weight)),'nonfinite SVC class weights')
      call require(all(class_weight>0),'SVC class weights must be positive')
      self%class_weight_=class_weight
    else if(self%class_weight_balanced)then
      do a=1,k
        self%class_weight_(a)=real(n,dp)/(k*real(count(encoded==a),dp))
      end do
    end if
    capacity=weights*(self%c*self%class_weight_(encoded))
    call require(all(ieee_is_finite(capacity)),'SVC capacity overflow')
    do a=1,k
      call require(any(capacity>0.and.encoded==a),'SVC every class needs positive capacity')
    end do
    self%x_fit_=x;self%gamma_=self%gamma
    if(self%gamma_==-1)then
      self%gamma_=1._dp/size(x,2)
      if(self%gamma_rule=='scale')then
        mean=sum(x)/size(x);variance=sum((x-mean)**2)/size(x);self%gamma_=1
        if(variance>0)self%gamma_=1/(size(x,2)*variance)
      end if
    end if
    gram=kernel_matrix(self,x,x);call require(all(ieee_is_finite(gram)),'nonfinite SVC Gram matrix')
    if(allocated(self%models_))deallocate(self%models_)
    allocate(self%models_(k*(k-1)/2));m=0;support=.false.;self%converged_=.true.;self%n_iter_=0
    self%n_iter_per_pair_=spread(0,1,size(self%models_));self%kkt_violation_=spread(0._dp,1,size(self%models_))
    self%intercept_=spread(0._dp,1,size(self%models_))
    do a=1,k-1
      do b=a+1,k
        m=m+1
        supports=pack([(i,i=1,n)],encoded==a.and.capacity>0)
        ids=pack([(i,i=1,n)],encoded==b.and.capacity>0);ids=[supports,ids]
        target=merge(-1._dp,1._dp,encoded(ids)==a)
        self%models_(m)%first=a;self%models_(m)%second=b
        call binary_fit(self%models_(m),gram(ids,ids),target,capacity(ids),self%tol,self%max_iter)
        target=self%models_(m)%dual;self%models_(m)%dual=spread(0._dp,1,n);self%models_(m)%dual(ids)=target
        support=support.or.abs(self%models_(m)%dual)>0
        self%converged_=self%converged_.and.self%models_(m)%converged
        self%n_iter_per_pair_(m)=self%models_(m)%n_iter;self%n_iter_=max(self%n_iter_,self%models_(m)%n_iter)
        self%intercept_(m)=-self%models_(m)%intercept;self%kkt_violation_(m)=self%models_(m)%kkt_violation
      end do
    end do
    self%support_original_=[integer::];self%n_support_=spread(0,1,k)
    do a=1,k
      supports=pack([(i,i=1,n)],support.and.encoded==a)
      self%n_support_(a)=size(supports);self%support_original_=[self%support_original_,supports]
    end do
    positions=spread(0,1,n);t=0
    do i=1,n
      if(weights(i)>0)then
        t=t+1;positions(i)=t
      end if
    end do
    self%support_=positions(self%support_original_)
    self%support_vectors_=x(self%support_original_,:)
    self%dual_coef_=spread(spread(0._dp,1,k-1),2,size(self%support_));m=0
    do a=1,k-1
      do b=a+1,k
        m=m+1
        do t=1,size(self%support_)
          i=self%support_original_(t)
          if(encoded(i)==a)self%dual_coef_(b-1,t)=-self%models_(m)%dual(i)
          if(encoded(i)==b)self%dual_coef_(a,t)=-self%models_(m)%dual(i)
        end do
      end do
    end do
    if(k==2)then
      self%dual_coef_=-self%dual_coef_;self%intercept_=-self%intercept_
    end if
    if(allocated(self%coef_))deallocate(self%coef_)
    if(self%kernel=='linear')then
      allocate(self%coef_(size(self%models_),size(x,2)))
      do m=1,size(self%models_)
        self%coef_(m,:)=matmul(self%models_(m)%dual,x)
      end do
      if(k>2)self%coef_=-self%coef_
    end if
    self%n_features_in_=size(x,2);self%fit_status_=merge(0,1,self%converged_)
  end subroutine
  function pair_margins(self,x)result(z)
    class(svc),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:),gram(:,:)
    integer::m
    call require(allocated(self%models_),'SVC is not fitted');call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SVC feature count mismatch')
    gram=kernel_matrix(self,x,self%x_fit_);allocate(z(size(x,1),size(self%models_)))
    do m=1,size(self%models_)
      z(:,m)=matmul(gram,self%models_(m)%dual)+self%models_(m)%intercept
    end do
  end function
  function ovr_scores(self,pairs)result(z)
    class(svc),intent(in)::self
    real(dp),intent(in)::pairs(:,:)
    real(dp),allocatable::z(:,:),confidence(:,:)
    integer::m,a,b
    allocate(z(size(pairs,1),size(self%classes_)),confidence(size(pairs,1),size(self%classes_)));z=0;confidence=0
    do m=1,size(self%models_)
      a=self%models_(m)%first;b=self%models_(m)%second
      z(:,a)=z(:,a)+merge(1._dp,0._dp,pairs(:,m)<=0);z(:,b)=z(:,b)+merge(1._dp,0._dp,pairs(:,m)>0)
      confidence(:,a)=confidence(:,a)-pairs(:,m);confidence(:,b)=confidence(:,b)+pairs(:,m)
    end do
    z=z+confidence/(3*(abs(confidence)+1))
  end function
  function svc_decision(self,x)result(z)
    class(svc),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:),pairs(:,:)
    call shape_controls(self);pairs=pair_margins(self,x)
    if(size(self%classes_)==2)then
      z=pairs
    else if(self%decision_function_shape=='ovo')then
      z=-pairs
    else
      z=ovr_scores(self,pairs)
    end if
  end function
  function svc_predict(self,x)result(y)
    class(svc),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer::y(size(x,1)),i,m,winner
    integer,allocatable::votes(:,:)
    real(dp),allocatable::z(:,:),scores(:,:)
    call shape_controls(self)
    call require(.not.(self%break_ties.and.self%decision_function_shape=='ovo'),'SVC break_ties requires ovr decision shape')
    z=pair_margins(self,x)
    if(self%break_ties.and.size(self%classes_)>2)then
      scores=ovr_scores(self,z);y=self%classes_(maxloc(scores,dim=2));return
    end if
    allocate(votes(size(x,1),size(self%classes_)));votes=0
    do i=1,size(x,1)
      do m=1,size(self%models_)
        winner=self%models_(m)%first
        if(z(i,m)>=0)winner=self%models_(m)%second
        votes(i,winner)=votes(i,winner)+1
      end do
      y(i)=self%classes_(maxloc(votes(i,:),dim=1))
    end do
  end function
  real(dp) function svc_score(self,x,y,sample_weight) result(v)
    class(svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    v=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function
end module
