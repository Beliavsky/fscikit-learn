! NuSVC formulation and scaling: sklearn 1.6.1 and bundled LIBSVM (BSD notices retained).
module sklearn_nu_classification
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel,accuracy_score
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_libsvm_sparse,only:svm_sorted,sparse_rows,libsvm_sparse_gram
  use sklearn_stateless_preprocessing,only:sorted_sparse
  use,intrinsic::iso_fortran_env,only:real32
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: nu_svc
  type :: nu_svc
    real(dp) :: nu=.5_dp,gamma=-1,gamma_=0,coef0=0,tol=1.e-3_dp
    integer :: degree=3,max_iter=200000,n_features_in_=0,fit_status_=1
    character(16) :: kernel='rbf',gamma_rule='scale',decision_function_shape='ovr'
    logical :: break_ties=.false.,converged_=.false.,shrinking=.true.
    integer, allocatable :: classes_(:),support_(:),n_support_(:),n_iter_(:)
    real(dp), allocatable :: support_vectors_(:,:),dual_coef_(:,:),coef_(:,:),intercept_(:)
    real(dp), allocatable :: pair_alpha_(:,:),pair_scales_(:),kkt_violation_(:)
    real(dp), allocatable, private :: x_fit(:,:),dual_full(:,:)
    type(csr_matrix),allocatable::x_fit_csr_,support_vectors_csr_,coef_csr_
    logical::sparse_fit_=.false.
    integer, allocatable, private :: first_class(:),second_class(:)
    character(16), private :: fitted_kernel='rbf'
    real(dp), private :: fitted_coef0=0
    integer, private :: fitted_degree=3
  contains
    procedure,private::fit_dense=>fit,fit_csr
    generic::fit=>fit_dense,fit_csr
    procedure,private::decision_dense=>decision_function,decision_csr
    generic::decision_function=>decision_dense,decision_csr
    procedure,private::predict_dense=>predict,predict_csr
    generic::predict=>predict_dense,predict_csr
    procedure,private::score_dense=>score,score_csr
    generic::score=>score_dense,score_csr
  end type
contains
  function kernel_matrix(self,x,y) result(gram)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: gram(:,:)
    select case(trim(self%fitted_kernel))
    case('linear'); gram=linear_kernel(x,y)
    case('rbf'); gram=rbf_kernel(x,y,self%gamma_)
    case('poly','polynomial'); gram=polynomial_kernel(x,y,self%fitted_degree,self%gamma_,self%fitted_coef0)
    case('sigmoid'); gram=sigmoid_kernel(x,y,self%gamma_,self%fitted_coef0)
    case('precomputed'); gram=x
    end select
  end function
  subroutine optimize(gram,labels,c,nu,tol,max_iter,shrinking,alpha_out,dual,bias,r,gap,iterations,converged)
    real(dp),intent(in)::gram(:,:),labels(:),c(:),nu,tol
    integer,intent(in)::max_iter
    logical,intent(in)::shrinking
    real(dp),intent(out)::alpha_out(:),dual(:),bias,r,gap
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp)::alpha(size(labels)),g(size(labels)),q(size(labels),size(labels)),y(size(labels)),qd(size(labels))
    real(dp)::gmax,gmax2,best,diff,quad,obj,oldi,oldj,delta,total,ci,cj,di,dj,rho,ub,lb
    integer::n,i,j,k,t,last,free_count,active,counter,order(size(labels))
    real(dp)::gbar(size(labels)),boxes(size(labels)),restored(size(labels))
    logical::unshrink,ui,uj
    real(dp)::mass,remaining,levels(2),maxima(4),hi,lo
    integer::h,ip,ineg
    real(dp)::gmaxn,gmaxn2
    ! LIBSVM nu-SVC two-group second-order selection, active-set shrinking and box updates.
    ! Its float Q cache is distinct from the double diagonal and query kernels.
    n=size(labels);y=labels;alpha=0;g=0;gbar=0;boxes=c
    order=[(i,i=1,n)];active=n;counter=min(n,1000)+1;unshrink=.false.
    do j=1,n
      qd(j)=gram(j,j)
      do i=1,n
        ! Column j is Source get_Q(j,:), including precomputed row orientation.
        q(i,j)=real(real(y(i)*y(j)*gram(j,i),real32),dp)
      end do
    end do
    mass=0
    do i=1,n
      mass=mass+nu*c(i)
    end do
    mass=mass/2
    do h=1,2
      remaining=mass
      do i=1,n
        if((y(i)>0).neqv.(h==1))cycle
        alpha(i)=min(c(i),remaining);remaining=remaining-alpha(i)
      end do
    end do
    do i=1,n
      if(alpha(i)<=0)cycle
      do j=1,n
        g(j)=g(j)+alpha(i)*q(j,i)
        if(alpha(i)>=c(i))gbar(j)=gbar(j)+c(i)*q(j,i)
      end do
    end do
    iterations=0;converged=.false.
    do while(iterations<max_iter)
      counter=counter-1
      if(counter==0)then
        counter=min(n,1000)
        if(shrinking)call shrink()
      end if
      block
        logical::optimal
        call select_pair(optimal)
        if(optimal)then
          call reconstruct();active=n
          call select_pair(optimal)
          if(optimal)then
            converged=.true.;exit
          end if
          counter=1
        end if
      end block
      j=last;iterations=iterations+1;ci=boxes(i);cj=boxes(j);oldi=alpha(i);oldj=alpha(j)
      quad=qd(order(i))+qd(order(j))-2*q(order(j),order(i));if(quad<=0)quad=1.e-12_dp
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
      di=alpha(i)-oldi;dj=alpha(j)-oldj
      do k=1,active
        ! Match C++ G[k] += (Q_i[k]*di + Q_j[k]*dj), preserving rounding.
        g(k)=g(k)+(q(order(k),order(i))*di+q(order(k),order(j))*dj)
      end do
      ui=oldi>=ci;uj=oldj>=cj
      if(ui.neqv.(alpha(i)>=ci))then
        do k=1,n
          if(ui)then
            gbar(k)=gbar(k)-ci*q(order(k),order(i))
          else
            gbar(k)=gbar(k)+ci*q(order(k),order(i))
          end if
        end do
      end if
      if(uj.neqv.(alpha(j)>=cj))then
        do k=1,n
          if(uj)then
            gbar(k)=gbar(k)-cj*q(order(k),order(j))
          else
            gbar(k)=gbar(k)+cj*q(order(k),order(j))
          end if
        end do
      end if
    end do
    do h=1,2
      ub=huge(1._dp);lb=-huge(1._dp);total=0;free_count=0
      do i=1,active
        if((y(i)>0).neqv.(h==1))cycle
        if(alpha(i)>=boxes(i))then
          lb=max(lb,g(i))
        else if(alpha(i)<=0)then
          ub=min(ub,g(i))
        else
          free_count=free_count+1;total=total+g(i)
        end if
      end do
      if(free_count>0)then
        levels(h)=total/free_count
      else
        call require(ub/=huge(1._dp).and.lb/=-huge(1._dp),'NuSVC unbounded margin normalization')
        levels(h)=(ub+lb)/2
      end if
    end do
    r=(levels(1)+levels(2))/2;rho=(levels(1)-levels(2))/2
    call require(ieee_is_finite(r),'NuSVC nonfinite margin normalization')
    call require(abs(r)>tiny(1._dp),'NuSVC zero margin normalization')
    restored(order)=alpha;alpha_out=restored
    dual=alpha_out*(labels/r);bias=-rho/r
    call require(all(ieee_is_finite(dual)).and.ieee_is_finite(bias),'NuSVC nonfinite coefficients')
    ! Keep the public residual on the original double kernel, independently of
    ! the float-cache stopping test and the active-set cap status.
    g=labels*matmul(gram,alpha_out*labels);gap=0
    do h=1,2
      hi=-huge(1._dp);lo=huge(1._dp)
      do i=1,n
        if((labels(i)>0).neqv.(h==1))cycle
        if(alpha_out(i)<c(i))lo=min(lo,g(i))
        if(alpha_out(i)>0)hi=max(hi,g(i))
      end do
      if(lo/=huge(1._dp).and.hi/=-huge(1._dp))gap=max(gap,hi-lo)
    end do
  contains
    subroutine select_pair(optimal)
      logical,intent(out)::optimal
      gmax=-huge(1._dp);gmax2=-huge(1._dp);gmaxn=-huge(1._dp);gmaxn2=-huge(1._dp)
      ip=0;ineg=0;last=0;best=huge(1._dp)
      do t=1,active
        if(y(t)>0)then
          if(alpha(t)<boxes(t).and.-g(t)>=gmax)then
            gmax=-g(t);ip=t
          end if
        else
          if(alpha(t)>0.and.g(t)>=gmaxn)then
            gmaxn=g(t);ineg=t
          end if
        end if
      end do
      do j=1,active
        if(y(j)>0)then
          if(alpha(j)<=0)cycle
          diff=gmax+g(j);gmax2=max(gmax2,g(j));i=ip
        else
          if(alpha(j)>=boxes(j))cycle
          diff=gmaxn-g(j);gmaxn2=max(gmaxn2,-g(j));i=ineg
        end if
        if(diff>0.and.i>0)then
          quad=qd(order(i))+qd(order(j))-2*q(order(j),order(i))
          if(quad<=0)quad=1.e-12_dp
          obj=-(diff*diff)/quad
          if(obj<=best)then
            best=obj;last=j
          end if
        end if
      end do
      ! Use guarded sums in place of C++ -INF for a fully bound group.
      hi=-huge(1._dp);lo=-huge(1._dp)
      if(ip>0.and.gmax2/=-huge(1._dp))hi=gmax+gmax2
      if(ineg>0.and.gmaxn2/=-huge(1._dp))lo=gmaxn+gmaxn2
      optimal=max(hi,lo)<tol.or.last==0
      if(.not.optimal)then
        i=ineg
        if(y(last)>0)i=ip
      end if
    end subroutine
    subroutine reconstruct()
      integer::a,b,nfree
      if(active==n)return
      g(active+1:)=gbar(active+1:)
      nfree=count(alpha(:active)>0.and.alpha(:active)<boxes(:active))
      if(real(nfree,dp)*n>2._dp*active*(n-active))then
        do a=active+1,n
          do b=1,active
            if(alpha(b)>0.and.alpha(b)<boxes(b)) &
              g(a)=g(a)+alpha(b)*q(order(b),order(a))
          end do
        end do
      else
        do a=1,active
          if(alpha(a)<=0.or.alpha(a)>=boxes(a))cycle
          do b=active+1,n
            g(b)=g(b)+alpha(a)*q(order(b),order(a))
          end do
        end do
      end if
    end subroutine
    logical function be_shrunk(a)result(v)
      integer,intent(in)::a
      v=.false.
      if(alpha(a)>=boxes(a))then
        if(y(a)>0)then
          v=-g(a)>maxima(1)
        else
          v=-g(a)>maxima(4)
        end if
      else if(alpha(a)<=0)then
        if(y(a)>0)then
          v=g(a)>maxima(2)
        else
          v=g(a)>maxima(3)
        end if
      end if
    end function
    subroutine shrink()
      integer::a
      real(dp)::v1,v2
      maxima=-huge(1._dp)
      do a=1,active
        if(y(a)>0)then
          if(alpha(a)<boxes(a))maxima(1)=max(maxima(1),-g(a))
          if(alpha(a)>0)maxima(2)=max(maxima(2),g(a))
        else
          if(alpha(a)<boxes(a))maxima(4)=max(maxima(4),-g(a))
          if(alpha(a)>0)maxima(3)=max(maxima(3),g(a))
        end if
      end do
      v1=-huge(1._dp);v2=v1
      if(all(maxima(:2)/=-huge(1._dp)))v1=maxima(1)+maxima(2)
      if(all(maxima(3:)/=-huge(1._dp)))v2=maxima(3)+maxima(4)
      if(.not.unshrink.and.max(v1,v2)<=tol*10)then
        unshrink=.true.;call reconstruct();active=n
      end if
      a=1
      do while(a<=active)
        if(be_shrunk(a))then
          active=active-1
          do while(active>=a)
            if(.not.be_shrunk(active+1))then
              call swap_index(a,active+1);exit
            end if
            active=active-1
          end do
        end if
        a=a+1
      end do
    end subroutine
    subroutine swap_index(a,b)
      integer,intent(in)::a,b
      integer::it
      real(dp)::v
      it=order(a);order(a)=order(b);order(b)=it
      v=y(a);y(a)=y(b);y(b)=v
      v=g(a);g(a)=g(b);g(b)=v
      v=gbar(a);gbar(a)=gbar(b);gbar(b)=v
      v=alpha(a);alpha(a)=alpha(b);alpha(b)=v
      v=boxes(a);boxes(a)=boxes(b);boxes(b)=v
    end subroutine
  end subroutine
  subroutine check_csr(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'NuSVC CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'NuSVC CSR input must be finite')
  end subroutine
  subroutine fit(self,x,y,sample_weight)
    class(nu_svc),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call check_xy(x,size(y));call fit_core(self,y,sample_weight,dense=x)
  end subroutine
  subroutine fit_csr(self,x,y,sample_weight)
    class(nu_svc),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call check_csr(x);call require(x%n_rows==size(y),'NuSVC CSR target count mismatch')
    call fit_core(self,y,sample_weight,sparse=x)
  end subroutine
  subroutine fit_core(self,y,sample_weight,dense,sparse)
    class(nu_svc),intent(inout)::self
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    type(csr_matrix)::canonical
    real(dp), allocatable :: gram(:,:),weights(:),mass(:),labels(:),alpha(:),dual(:)
    real(dp) :: mean,variance,total,bias,r,gap,entries
    integer, allocatable :: ids(:),left(:),right(:),supports(:)
    logical :: active(size(y)),pair_converged
    integer :: n,p,k,pairs,i,j,a,b,m,t,iterations,row
    n=size(y)
    if(present(sparse))then
      p=sparse%n_cols
    else
      p=size(dense,2)
    end if
    call require(ieee_is_finite(self%nu),'NuSVC nu must be finite and in (0,1]')
    call require(self%nu>0.and.self%nu<=1,'NuSVC nu must be finite and in (0,1]')
    call require(ieee_is_finite(self%tol),'NuSVC tolerance must be finite and positive')
    call require(self%tol>0.and.self%max_iter>0,'NuSVC tolerance and max_iter must be positive')
    call require(any(self%kernel==[character(16)::'linear','rbf','poly','polynomial','sigmoid','precomputed']), &
      'unknown NuSVC kernel')
    call require(self%gamma_rule=='scale'.or.self%gamma_rule=='auto','unknown NuSVC gamma rule')
    call require(ieee_is_finite(self%gamma).and.ieee_is_finite(self%coef0),'nonfinite NuSVC kernel parameters')
    call require((self%gamma==-1.or.self%gamma>=0).and.self%degree>=0,'invalid NuSVC gamma or degree')
    call shape_controls(self)
    self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'NuSVC requires at least two classes')
    weights=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'NuSVC sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'NuSVC weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'NuSVC weights must be finite and nonnegative')
      weights=sample_weight
    end if
    total=sum(weights)
    call require(ieee_is_finite(total),'NuSVC weights must have positive finite sum')
    call require(total>0,'NuSVC weights must have positive finite sum')
    allocate(mass(k))
    do a=1,k
      mass(a)=sum(weights,mask=y==self%classes_(a))
    end do
    call require(all(mass>0),'NuSVC every class needs positive sample weight')
    do a=1,k-1
      do b=a+1,k
        call require(self%nu*(mass(a)/2+mass(b)/2)<=min(mass(a),mass(b)), &
          'NuSVC specified nu is infeasible')
      end do
    end do
    self%sparse_fit_=present(sparse)
    if(allocated(self%x_fit))deallocate(self%x_fit)
    if(allocated(self%x_fit_csr_))deallocate(self%x_fit_csr_)
    if(allocated(self%support_vectors_))deallocate(self%support_vectors_)
    if(allocated(self%support_vectors_csr_))deallocate(self%support_vectors_csr_)
    if(allocated(self%coef_csr_))deallocate(self%coef_csr_)
    self%gamma_=self%gamma
    self%fitted_kernel=self%kernel;self%fitted_coef0=self%coef0;self%fitted_degree=self%degree
    if(present(sparse))then
      call require(self%kernel/='precomputed','NuSVC precomputed inputs must be dense')
      self%x_fit_csr_=svm_sorted(sparse)
      if(self%gamma_==-1)then
        self%gamma_=1._dp/p
        if(self%gamma_rule=='scale')then
          canonical=sorted_sparse(sparse,.true.);entries=real(n,dp)*p
          mean=sum(canonical%data)/entries;variance=sum(canonical%data**2)/entries-mean**2;self%gamma_=1
          if(variance/=0)self%gamma_=1/(p*variance)
        end if
      end if
      gram=libsvm_sparse_gram(self%x_fit_csr_,self%x_fit_csr_,self%fitted_kernel,self%gamma_, &
        self%fitted_coef0,self%fitted_degree,.true.)
    else
      self%x_fit=dense
      if(self%gamma_==-1)then
        self%gamma_=1._dp/p
        if(self%gamma_rule=='scale')then
          mean=sum(dense)/size(dense);variance=sum((dense-mean)**2)/size(dense);self%gamma_=1
          if(variance>0)self%gamma_=1/(p*variance)
        end if
      end if
      if(self%kernel=='precomputed')then
        self%gamma_=0
        call require(p==n,'NuSVC precomputed training matrix must be square')
        call require(all(abs(dense-transpose(dense))<=1.e-10_dp*max(1._dp,abs(dense),abs(transpose(dense)))), &
          'NuSVC precomputed training matrix must be symmetric')
      end if
      gram=kernel_matrix(self,dense,dense)
    end if
    self%n_features_in_=p
    call require(all(ieee_is_finite(gram)),'NuSVC nonfinite Gram matrix')
    pairs=k*(k-1)/2
    if(allocated(self%dual_full)) then
      deallocate(self%dual_full,self%pair_alpha_,self%pair_scales_,self%intercept_,self%kkt_violation_, &
        self%n_iter_,self%first_class,self%second_class)
    end if
    if(allocated(self%coef_)) deallocate(self%coef_)
    allocate(self%dual_full(n,pairs),self%pair_alpha_(n,pairs),self%pair_scales_(pairs), &
      self%intercept_(pairs),self%kkt_violation_(pairs),self%n_iter_(pairs), &
      self%first_class(pairs),self%second_class(pairs))
    self%dual_full=0; self%pair_alpha_=0; active=.false.; m=0; self%converged_=.true.
    do a=1,k-1
      do b=a+1,k
        m=m+1; self%first_class(m)=a; self%second_class(m)=b
        left=pack([(i,i=1,n)],y==self%classes_(a).and.weights>0)
        right=pack([(i,i=1,n)],y==self%classes_(b).and.weights>0); ids=[left,right]
        labels=merge(1._dp,-1._dp,y(ids)==self%classes_(a)); t=size(ids)
        allocate(alpha(t),dual(t))
        call optimize(gram(ids,ids),labels,weights(ids),self%nu,self%tol,self%max_iter,self%shrinking,alpha,dual,bias,r,gap,iterations,pair_converged)
        self%pair_alpha_(ids,m)=alpha; self%dual_full(ids,m)=dual
        self%pair_scales_(m)=r; self%intercept_(m)=bias; self%kkt_violation_(m)=gap; self%n_iter_(m)=iterations
        active=active.or.abs(self%dual_full(:,m))>0
        self%converged_=self%converged_.and.pair_converged; deallocate(alpha,dual)
      end do
    end do
    supports=[integer::]; self%n_support_=spread(0,1,k)
    do a=1,k
      ids=pack([(i,i=1,n)],active.and.y==self%classes_(a)); self%n_support_(a)=size(ids); supports=[supports,ids]
    end do
    self%support_=supports
    if(present(sparse))then
      self%support_vectors_csr_=sparse_rows(self%x_fit_csr_,supports)
    else if(self%kernel=='precomputed') then
      self%support_vectors_=reshape([real(dp)::],[0,0])
    else
      self%support_vectors_=dense(supports,:)
    end if
    self%dual_coef_=spread(spread(0._dp,1,k-1),2,size(supports))
    do m=1,pairs
      a=self%first_class(m); b=self%second_class(m)
      do j=1,size(supports)
        i=supports(j); row=0
        if(y(i)==self%classes_(a)) row=b-1
        if(y(i)==self%classes_(b)) row=a
        if(row>0) self%dual_coef_(row,j)=self%dual_full(i,m)
      end do
    end do
    if(k==2) then
      self%dual_coef_=-self%dual_coef_; self%intercept_=-self%intercept_; self%dual_full=-self%dual_full
    end if
    if(self%kernel=='linear')then
      if(present(sparse))then
        call sparse_linear_coef(self)
      else
        self%coef_=matmul(transpose(self%dual_full),dense)
      end if
    end if
    self%fit_status_=merge(0,1,self%converged_)
  end subroutine
  subroutine sparse_linear_coef(self)
    class(nu_svc),intent(inout)::self
    type(csr_matrix)::row,canonical
    integer::m,i,k,t,nz
    real(dp)::weight
    allocate(self%coef_csr_)
    self%coef_csr_%n_rows=size(self%dual_full,2);self%coef_csr_%n_cols=self%x_fit_csr_%n_cols
    allocate(self%coef_csr_%indptr(size(self%dual_full,2)+1));self%coef_csr_%indptr(1)=1
    self%coef_csr_%indices=[integer::];self%coef_csr_%data=[real(dp)::]
    row%n_rows=1;row%n_cols=self%x_fit_csr_%n_cols
    allocate(row%indptr(2),row%indices(size(self%x_fit_csr_%data)),row%data(size(self%x_fit_csr_%data)))
    do m=1,size(self%dual_full,2)
      t=0
      do i=1,self%x_fit_csr_%n_rows
        weight=self%dual_full(i,m)
        if(weight==0)cycle
        do k=self%x_fit_csr_%indptr(i),self%x_fit_csr_%indptr(i+1)-1
          t=t+1;row%indices(t)=self%x_fit_csr_%indices(k);row%data(t)=weight*self%x_fit_csr_%data(k)
        end do
      end do
      row%indptr=[1,t+1]
      block
        type(csr_matrix)::used
        used%n_rows=1;used%n_cols=row%n_cols;used%indptr=row%indptr
        used%indices=row%indices(:t);used%data=row%data(:t);canonical=sorted_sparse(used,.true.)
      end block
      nz=count(canonical%data/=0)
      self%coef_csr_%indices=[self%coef_csr_%indices,pack(canonical%indices,canonical%data/=0)]
      self%coef_csr_%data=[self%coef_csr_%data,pack(canonical%data,canonical%data/=0)]
      self%coef_csr_%indptr(m+1)=self%coef_csr_%indptr(m)+nz
    end do
  end subroutine
  subroutine shape_controls(self)
    class(nu_svc), intent(in) :: self
    call require(self%decision_function_shape=='ovo'.or.self%decision_function_shape=='ovr', &
      'NuSVC decision_function_shape must be ovo or ovr')
    call require(.not.(self%break_ties.and.self%decision_function_shape=='ovo'), &
      'NuSVC break_ties requires ovr decision scores')
  end subroutine
  function pair_scores(self,x) result(z)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),gram(:,:)
    integer :: m
    type(csr_matrix)::query
    call require(allocated(self%dual_full),'NuSVC is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'NuSVC query feature count mismatch')
    call shape_controls(self)
    if(self%sparse_fit_)then
      query=csr_from_dense(x)
      gram=libsvm_sparse_gram(query,self%x_fit_csr_,self%fitted_kernel,self%gamma_, &
        self%fitted_coef0,self%fitted_degree,.false.)
    else
      gram=kernel_matrix(self,x,self%x_fit)
    end if
    z=matmul(gram,self%dual_full)
    do m=1,size(z,2)
      z(:,m)=z(:,m)+self%intercept_(m)
    end do
  end function
  function pair_scores_csr(self,x)result(z)
    class(nu_svc),intent(in)::self
    type(csr_matrix),intent(in)::x
    type(csr_matrix)::query
    real(dp),allocatable::z(:,:),gram(:,:)
    integer::m
    call require(allocated(self%dual_full),'NuSVC is not fitted');call check_csr(x)
    call require(self%sparse_fit_,'NuSVC CSR query requires a CSR fit')
    call require(x%n_cols==self%n_features_in_,'NuSVC query feature count mismatch')
    call shape_controls(self);query=svm_sorted(x)
    gram=libsvm_sparse_gram(query,self%x_fit_csr_,self%fitted_kernel,self%gamma_, &
      self%fitted_coef0,self%fitted_degree,.false.)
    z=matmul(gram,self%dual_full)
    do m=1,size(z,2)
      z(:,m)=z(:,m)+self%intercept_(m)
    end do
  end function
  function decision_function(self,x)result(z)
    class(nu_svc),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    z=decision_from_pairs(self,pair_scores(self,x))
  end function
  function decision_csr(self,x)result(z)
    class(nu_svc),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    z=decision_from_pairs(self,pair_scores_csr(self,x))
  end function
  function decision_from_pairs(self,raw) result(z)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: raw(:,:)
    real(dp), allocatable :: z(:,:),confidence(:,:)
    integer :: i,m,a,b,k
    k=size(self%classes_)
    if(k==2.or.self%decision_function_shape=='ovo') then
      z=raw
      return
    end if
    allocate(z(size(raw,1),k),confidence(size(raw,1),k)); z=0; confidence=0
    do m=1,size(raw,2)
      a=self%first_class(m); b=self%second_class(m)
      confidence(:,a)=confidence(:,a)+raw(:,m); confidence(:,b)=confidence(:,b)-raw(:,m)
      do i=1,size(raw,1)
        if(raw(i,m)>=0) then
          z(i,a)=z(i,a)+1
        else
          z(i,b)=z(i,b)+1
        end if
      end do
    end do
    z=z+confidence/(3*(abs(confidence)+1))
  end function
  function predict(self,x)result(y)
    class(nu_svc),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer::y(size(x,1))
    y=prediction_from_pairs(self,pair_scores(self,x))
  end function
  function predict_csr(self,x)result(y)
    class(nu_svc),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::y(:)
    y=prediction_from_pairs(self,pair_scores_csr(self,x))
  end function
  function prediction_from_pairs(self,raw)result(y)
    class(nu_svc),intent(in)::self
    real(dp),intent(in)::raw(:,:)
    integer,allocatable::y(:),votes(:,:)
    integer::i,m,a,b
    real(dp),allocatable::z(:,:)
    allocate(y(size(raw,1)))
    if(self%break_ties) then
      z=decision_from_pairs(self,raw)
      if(size(self%classes_)>2) then
        do i=1,size(y)
          y(i)=self%classes_(maxloc(z(i,:),dim=1))
        end do
        return
      end if
    end if
    z=raw
    if(size(self%classes_)==2) then
      y=merge(self%classes_(2),self%classes_(1),z(:,1)>=0)
      return
    end if
    allocate(votes(size(raw,1),size(self%classes_))); votes=0
    do m=1,size(z,2)
      a=self%first_class(m); b=self%second_class(m)
      do i=1,size(y)
        if(z(i,m)>0) then
          votes(i,a)=votes(i,a)+1
        else
          votes(i,b)=votes(i,b)+1
        end if
      end do
    end do
    do i=1,size(y)
      y(i)=self%classes_(maxloc(votes(i,:),dim=1))
    end do
  end function
  real(dp) function score(self,x,y,sample_weight) result(value)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp) function score_csr(self,x,y,sample_weight)result(value)
    class(nu_svc),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function
end module
