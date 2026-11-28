! Dense/CSR SVR, NuSVR and one-class SVM contracts from sklearn/svm/_base.py,
! _classes.py and its bundled LIBSVM. Native most-violating-pair optimization.
! See THIRD_PARTY_NOTICES.md and LICENSES/libsvm-COPYRIGHT for attribution.
module sklearn_kernel_svm
  use sklearn_common
  use sklearn_metrics, only: rbf_kernel,linear_kernel,polynomial_kernel,r2_score
  use sklearn_sparse, only:csr_matrix,csr_from_dense
  use sklearn_stateless_preprocessing, only:sorted_sparse
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: svr,nu_svr,one_class_svm
  type :: kernel_model
    character(16) :: kernel='rbf',gamma_rule='scale'
    real(dp) :: gamma=-1,gamma_=0,coef0=0,tol=1.e-3_dp,intercept_=0,kkt_violation_=0
    integer :: degree=3,max_iter=100000,n_iter_=0,fit_status_=1,n_support_=0
    logical :: converged_=.false.
    logical :: sparse_fit_=.false.
    type(csr_matrix), allocatable :: x_fit_csr_,support_vectors_csr_
    integer, allocatable :: support_(:)
    real(dp), allocatable :: support_vectors_(:,:),dual_coef_(:),coef_(:),x_fit_(:,:)
    real(dp), allocatable, private :: dual_full_(:)
    character(16), private :: kernel_fit_='rbf'
    integer, private :: degree_fit_=3
    real(dp), private :: coef0_fit_=0
  contains
    procedure,private :: decision_dense=>kernel_decision,decision_csr=>kernel_decision_csr
    generic :: decision_function=>decision_dense,decision_csr
  end type
  type, extends(kernel_model) :: regression_model
    real(dp) :: c=1
  contains
    procedure,private :: predict_dense=>regression_predict,predict_csr=>regression_predict_csr
    generic :: predict=>predict_dense,predict_csr
    procedure,private :: score_dense=>regression_score,score_csr=>regression_score_csr
    generic :: score=>score_dense,score_csr
  end type
  type, extends(regression_model) :: svr
    real(dp) :: epsilon=0.1_dp
  contains
    procedure,private :: fit_dense=>svr_fit,fit_csr=>svr_fit_csr
    generic :: fit=>fit_dense,fit_csr
  end type
  type, extends(regression_model) :: nu_svr
    real(dp) :: nu=0.5_dp,epsilon_=0
  contains
    procedure,private :: fit_dense=>nusvr_fit,fit_csr=>nusvr_fit_csr
    generic :: fit=>fit_dense,fit_csr
  end type
  type, extends(kernel_model) :: one_class_svm
    real(dp) :: nu=0.5_dp,offset_=0
  contains
    procedure,private :: fit_dense=>one_fit,fit_csr=>one_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure,private :: predict_dense=>one_predict,predict_csr=>one_predict_csr
    generic :: predict=>predict_dense,predict_csr
    procedure,private :: fit_predict_dense=>one_fit_predict,fit_predict_csr=>one_fit_predict_csr
    generic :: fit_predict=>fit_predict_dense,fit_predict_csr
    procedure,private :: score_samples_dense=>one_score_samples,score_samples_csr=>one_score_samples_csr
    generic :: score_samples=>score_samples_dense,score_samples_csr
  end type
contains
  function gram_matrix(self,x,y) result(k)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(self%kernel_fit_)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('linear'); k=linear_kernel(x,y)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree_fit_,self%gamma_,self%coef0_fit_)
    case('precomputed'); k=x
    end select
  end function
  subroutine setup_controls(self)
    class(kernel_model), intent(inout) :: self
    call require(self%max_iter>0,'kernel SVM max_iter must be positive')
    call require(ieee_is_finite(self%tol),'invalid kernel SVM tolerance')
    call require(self%tol>0,'invalid kernel SVM tolerance')
    call require(self%kernel=='rbf'.or.self%kernel=='linear'.or.self%kernel=='poly'.or. &
      self%kernel=='polynomial'.or.self%kernel=='precomputed','unsupported kernel SVM kernel')
    call require(ieee_is_finite(self%gamma).and.ieee_is_finite(self%coef0),'invalid kernel SVM parameters')
    call require((self%gamma>=0.or.self%gamma==-1).and.self%degree>=0.and.self%coef0>=0, &
      'invalid kernel SVM parameters')
    call require(self%gamma_rule=='scale'.or.self%gamma_rule=='auto','invalid kernel SVM gamma rule')
    self%kernel_fit_=self%kernel; self%degree_fit_=self%degree; self%coef0_fit_=self%coef0
    if(allocated(self%coef_)) deallocate(self%coef_)
    if(allocated(self%support_vectors_csr_))deallocate(self%support_vectors_csr_)
  end subroutine
  subroutine setup(self,x,gram)
    class(kernel_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: gram(:,:)
    real(dp), allocatable :: values(:),vectors(:,:)
    real(dp) :: variance,mean
    integer :: n
    call check_x(x); n=size(x,1);call setup_controls(self)
    self%sparse_fit_=.false.
    if(allocated(self%x_fit_csr_))deallocate(self%x_fit_csr_)
    self%x_fit_=x; self%gamma_=self%gamma
    if(self%gamma_==-1) then
      self%gamma_=1._dp/size(x,2)
      if(self%gamma_rule=='scale') then
        mean=sum(x)/size(x); variance=sum((x-mean)**2)/size(x); self%gamma_=1
        if(variance>0) self%gamma_=1/(size(x,2)*variance)
      end if
    end if
    if(self%kernel_fit_=='precomputed') then
      self%gamma_=0
      call require(size(x,2)==n,'kernel SVM precomputed matrix must be square')
      call require(all(abs(x-transpose(x))<=1.e-10_dp*max(1._dp,abs(x),abs(transpose(x)))), &
        'kernel SVM precomputed matrix must be symmetric')
      allocate(values(n),vectors(n,n)); call symmetric_eigen(x,values,vectors)
      call require(minval(values)>=-1.e-10_dp*max(1._dp,maxval(abs(values))), &
        'kernel SVM precomputed matrix must be positive semidefinite')
    end if
    gram=gram_matrix(self,x,x)
    call require(all(ieee_is_finite(gram)),'kernel SVM nonfinite Gram matrix')
  end subroutine
  subroutine check_csr(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'kernel SVM CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'kernel SVM CSR input must be finite')
  end subroutine
  subroutine setup_csr(self,x,gram)
    class(kernel_model),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable,intent(out)::gram(:,:)
    type(csr_matrix)::canonical
    real(dp)::mean,variance,mass
    call check_csr(x);call setup_controls(self)
    call require(self%kernel_fit_/='precomputed','kernel SVM precomputed inputs must be dense')
    self%x_fit_csr_=svm_sorted(x);self%sparse_fit_=.true.
    if(allocated(self%x_fit_))deallocate(self%x_fit_)
    if(allocated(self%support_vectors_))deallocate(self%support_vectors_)
    self%gamma_=self%gamma
    if(self%gamma_==-1)then
      self%gamma_=1._dp/x%n_cols
      if(self%gamma_rule=='scale')then
        ! SciPy multiply/mean coalesce duplicates for the second moment.
        canonical=sorted_sparse(x,.true.);mass=real(x%n_rows,dp)*real(x%n_cols,dp)
        mean=sum(canonical%data)/mass
        variance=sum(canonical%data**2)/mass-mean**2;self%gamma_=1
        if(variance/=0)self%gamma_=1/(x%n_cols*variance)
      end if
    end if
    gram=sparse_gram(self,self%x_fit_csr_,self%x_fit_csr_,.true.)
    call require(all(ieee_is_finite(gram)),'kernel SVM nonfinite Gram matrix')
  end subroutine
  real(dp) function sparse_inner(x,i,y,j,distance) result(value)
    type(csr_matrix),intent(in)::x,y
    integer,intent(in)::i,j
    logical,intent(in)::distance
    integer::a,b,ae,be
    value=0;a=x%indptr(i);ae=x%indptr(i+1);b=y%indptr(j);be=y%indptr(j+1)
    do while(a<ae.and.b<be)
      if(x%indices(a)==y%indices(b))then
        if(distance)then
          value=value+(x%data(a)-y%data(b))**2
        else
          value=value+x%data(a)*y%data(b)
        end if
        a=a+1;b=b+1
      else if(x%indices(a)<y%indices(b))then
        if(distance)value=value+x%data(a)**2
        a=a+1
      else
        if(distance)value=value+y%data(b)**2
        b=b+1
      end if
    end do
    if(distance)then
      do while(a<ae)
        value=value+x%data(a)**2;a=a+1
      end do
      do while(b<be)
        value=value+y%data(b)**2;b=b+1
      end do
    end if
  end function
  function svm_sorted(x) result(y)
    type(csr_matrix),intent(in)::x
    type(csr_matrix)::y
    integer::i,a,b,j,k,key,depth,length
    real(dp)::value
    logical::ordered
    y=x;ordered=.true.
    do i=1,x%n_rows
      a=x%indptr(i);b=x%indptr(i+1)-1
      if(any(x%indices(a+1:b)<x%indices(a:b-1)))ordered=.false.
    end do
    if(ordered)return
    ! SciPy csr_sort_indices sorts every row with std::sort when any row is
    ! unordered. Preserve the reference wheel's median/partition tie order.
    do i=1,x%n_rows
      a=x%indptr(i);b=x%indptr(i+1);length=b-a;depth=0
      do while(length>1)
        length=length/2;depth=depth+2
      end do
      call partition_sort(a,b,depth)
      do j=a+1,b-1
        key=y%indices(j);value=y%data(j);k=j
        do while(k>a)
          if(y%indices(k-1)<=key)exit
          y%indices(k)=y%indices(k-1);y%data(k)=y%data(k-1);k=k-1
        end do
        y%indices(k)=key;y%data(k)=value
      end do
    end do
  contains
    subroutine swap(a,b)
      integer,intent(in)::a,b
      integer::key
      real(dp)::value
      key=y%indices(a);y%indices(a)=y%indices(b);y%indices(b)=key
      value=y%data(a);y%data(a)=y%data(b);y%data(b)=value
    end subroutine
    recursive subroutine partition_sort(lo,hi,depth)
      integer,intent(in)::lo,hi,depth
      integer::a,b,c,p,left,right,start,last
      if(hi-lo<=16)return
      if(depth==0)then
        ! Bounded-memory heap fallback for adversarial partition sequences.
        do start=(hi-lo)/2,1,-1
          call sift(start,hi-lo,lo)
        end do
        do last=hi-lo,2,-1
          call swap(lo,lo+last-1);call sift(1,last-1,lo)
        end do
        return
      end if
      a=lo+1;b=lo+(hi-lo)/2;c=hi-1
      if(y%indices(a)<y%indices(b))then
        if(y%indices(b)<y%indices(c))then
          p=b
        else if(y%indices(a)<y%indices(c))then
          p=c
        else
          p=a
        end if
      else
        if(y%indices(a)<y%indices(c))then
          p=a
        else if(y%indices(b)<y%indices(c))then
          p=c
        else
          p=b
        end if
      end if
      call swap(lo,p);left=lo+1;right=hi
      do
        do while(y%indices(left)<y%indices(lo))
          left=left+1
        end do
        right=right-1
        do while(y%indices(lo)<y%indices(right))
          right=right-1
        end do
        if(left>=right)exit
        call swap(left,right);left=left+1
      end do
      call partition_sort(left,hi,depth-1);call partition_sort(lo,left,depth-1)
    end subroutine
    subroutine sift(first,last,base)
      integer,intent(in)::first,last,base
      integer::root,child
      root=first
      do while(2*root<=last)
        child=2*root
        if(child<last)then
          if(y%indices(base+child-1)<=y%indices(base+child))child=child+1
        end if
        if(y%indices(base+root-1)>=y%indices(base+child-1))exit
        call swap(base+root-1,base+child-1);root=child
      end do
    end subroutine
  end function
  function sparse_gram(self,x,y,training) result(k)
    class(kernel_model),intent(in)::self
    type(csr_matrix),intent(in)::x,y
    logical,intent(in)::training
    real(dp),allocatable::k(:,:)
    real(dp)::dot,distance
    integer::i,j
    allocate(k(x%n_rows,y%n_rows))
    do j=1,y%n_rows
      do i=1,x%n_rows
        dot=sparse_inner(x,i,y,j,.false.)
        select case(self%kernel_fit_)
        case('linear');k(i,j)=dot
        case('poly','polynomial');k(i,j)=(self%gamma_*dot+self%coef0_fit_)**self%degree_fit_
        case('rbf')
          if(training)then
            distance=sparse_inner(x,i,x,i,.false.)+sparse_inner(y,j,y,j,.false.)-2*dot
          else
            distance=sparse_inner(x,i,y,j,.true.)
          end if
          k(i,j)=exp(-self%gamma_*distance)
        end select
      end do
    end do
  end function
  function sparse_rows(x,rows) result(y)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::rows(:)
    type(csr_matrix)::y
    integer::i,a,b,offset
    y%n_rows=size(rows);y%n_cols=x%n_cols;allocate(y%indptr(size(rows)+1));y%indptr(1)=1
    do i=1,size(rows)
      y%indptr(i+1)=y%indptr(i)+x%indptr(rows(i)+1)-x%indptr(rows(i))
    end do
    allocate(y%indices(y%indptr(size(rows)+1)-1),y%data(y%indptr(size(rows)+1)-1))
    do i=1,size(rows)
      a=x%indptr(rows(i));b=x%indptr(rows(i)+1)-1;offset=y%indptr(i)
      y%indices(offset:offset+b-a)=x%indices(a:b);y%data(offset:offset+b-a)=x%data(a:b)
    end do
  end function
  function bounds(n,c,sample_weight) result(cap)
    integer, intent(in) :: n
    real(dp), intent(in) :: c
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: cap(n)
    call require(ieee_is_finite(c),'kernel SVM C must be positive')
    call require(c>0,'kernel SVM C must be positive'); cap=c
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'kernel SVM weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'invalid kernel SVM sample weights')
      call require(all(sample_weight>=0).and.any(sample_weight>0),'invalid kernel SVM sample weights')
      cap=c*sample_weight
    end if
    call require(all(ieee_is_finite(cap)),'kernel SVM capacity overflow')
  end function
  subroutine select_pair(g,alpha,cap,signs,index,group,epsilon,groups,first,last,gap)
    real(dp), intent(in) :: g(:),alpha(:),cap(:),signs(:),epsilon
    integer, intent(in) :: index(:),group(:),groups
    integer, intent(out) :: first,last
    real(dp), intent(out) :: gap
    real(dp) :: low(groups),high(groups),value
    integer :: left(groups),right(groups),j,h
    low=huge(1._dp); high=-huge(1._dp); left=0; right=0
    do j=1,size(alpha)
      if(cap(j)==0) cycle
      h=group(j); value=g(index(j))+signs(j)*epsilon
      if((signs(j)>0.and.alpha(j)<cap(j)).or.(signs(j)<0.and.alpha(j)>0)) then
        if(value<low(h)) then
          low(h)=value; left(h)=j
        end if
      end if
      if((signs(j)>0.and.alpha(j)>0).or.(signs(j)<0.and.alpha(j)<cap(j))) then
        if(value>high(h)) then
          high(h)=value; right(h)=j
        end if
      end if
    end do
    gap=0; first=0; last=0
    do h=1,groups
      if(left(h)==0.or.right(h)==0) cycle
      if(high(h)-low(h)>gap) then
        gap=high(h)-low(h); first=left(h); last=right(h)
      end if
    end do
  end subroutine
  subroutine working_pair(gram,g,alpha,cap,signs,index,group,epsilon,groups,first,last,best_step)
    real(dp), intent(in) :: gram(:,:),g(:),alpha(:),cap(:),signs(:),epsilon
    integer, intent(in) :: index(:),group(:),groups
    integer, intent(out) :: first,last
    real(dp), intent(out) :: best_step
    real(dp) :: low,value,inc_limit,dec_limit,eta,diff,step,gain,best_gain
    integer :: h,i,j,left
    first=0; last=0; best_step=0; best_gain=-1
    do h=1,groups
      low=huge(1._dp); left=0
      do i=1,size(alpha)
        if(group(i)/=h.or.cap(i)==0) cycle
        if(.not.((signs(i)>0.and.alpha(i)<cap(i)).or.(signs(i)<0.and.alpha(i)>0))) cycle
        value=g(index(i))+signs(i)*epsilon
        if(value<low) then
          low=value; left=i
        end if
      end do
      if(left==0) cycle
      inc_limit=alpha(left)
      if(signs(left)>0) inc_limit=cap(left)-alpha(left)
      do j=1,size(alpha)
        if(j==left.or.group(j)/=h.or.cap(j)==0) cycle
        if(.not.((signs(j)>0.and.alpha(j)>0).or.(signs(j)<0.and.alpha(j)<cap(j)))) cycle
        diff=g(index(j))+signs(j)*epsilon-low
        if(diff<=0) cycle
        dec_limit=cap(j)-alpha(j)
        if(signs(j)>0) dec_limit=alpha(j)
        step=min(inc_limit,dec_limit)
        eta=max(0._dp,gram(index(left),index(left))+gram(index(j),index(j))-2*gram(index(left),index(j)))
        if(eta>0) step=min(step,diff/eta)
        gain=step*(diff-eta*step/2)
        if(gain>best_gain) then
          first=left; last=j; best_step=step; best_gain=gain
        end if
      end do
    end do
  end subroutine
  subroutine optimize(self,gram,y,capacity,mode,epsilon,nu,levels)
    class(kernel_model), intent(inout) :: self
    real(dp), intent(in) :: gram(:,:),y(:),capacity(:),epsilon,nu
    integer, intent(in) :: mode ! 1: epsilon SVR; 2: nu SVR; 3: one-class
    real(dp), intent(out) :: levels(2)
    real(dp), allocatable :: alpha(:),cap(:),signs(:),g(:),beta(:)
    integer, allocatable :: index(:),group(:)
    real(dp) :: remaining,target,gap,step,low,high,free_sum,value,bound_tol,fallback_sum
    integer :: n,m,groups,i,j,a,b,it,h,count_free,fallback_count
    n=size(y); m=2*n; groups=1
    if(mode==3) m=n
    if(mode==2) groups=2
    allocate(alpha(m),cap(m),signs(m),g(n),beta(n),index(m),group(m)); alpha=0; signs=1; group=1
    index=[(1+mod(i-1,n),i=1,m)]; cap=capacity(index)
    if(mode/=3) signs(n+1:)=-1
    if(mode==2) group(n+1:)=2
    if(mode==3) then
      ! A feasible interior start preserves symmetry and avoids unnecessarily
      ! exciting very flat dual directions in smooth, symmetric kernels.
      alpha=nu*cap
    else if(mode==2) then
      target=nu*sum(capacity)
      target=target/2
      remaining=target
      do i=1,n
        alpha(i)=min(cap(i),max(0._dp,remaining)); remaining=remaining-alpha(i)
      end do
      alpha(n+1:)=alpha(:n)
    end if
    beta=0
    do i=1,m
      beta(index(i))=beta(index(i))+signs(i)*alpha(i)
    end do
    g=matmul(gram,beta)-y; self%n_iter_=0
    do it=1,self%max_iter
      call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,gap)
      if(gap<=self%tol) then
        beta=0
        do i=1,m
          beta(index(i))=beta(index(i))+signs(i)*alpha(i)
        end do
        g=matmul(gram,beta)-y
        call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,gap)
        if(gap<=self%tol) exit
      end if
      call working_pair(gram,g,alpha,cap,signs,index,group,epsilon,groups,a,b,step)
      if(a==0.or.b==0) exit
      i=index(a); j=index(b)
      if(step<=tiny(1._dp)) exit
      alpha(a)=max(0._dp,min(cap(a),alpha(a)+signs(a)*step))
      alpha(b)=max(0._dp,min(cap(b),alpha(b)-signs(b)*step))
      beta(i)=beta(i)+step; beta(j)=beta(j)-step
      g=g+step*(gram(:,i)-gram(:,j)); self%n_iter_=it
      if(mod(it,1000)==0) g=matmul(gram,beta)-y
    end do
    ! Reconstruct signed coefficients before final diagnostics to bound cache drift.
    beta=0
    do i=1,m
      beta(index(i))=beta(index(i))+signs(i)*alpha(i)
    end do
    g=matmul(gram,beta)-y
    call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,self%kkt_violation_)
    self%converged_=self%kkt_violation_<=self%tol; self%fit_status_=merge(0,1,self%converged_)
    levels=0
    do h=1,groups
      low=huge(1._dp); high=-huge(1._dp); free_sum=0; count_free=0; fallback_sum=0; fallback_count=0
      do i=1,m
        if(group(i)/=h.or.cap(i)==0) cycle
        value=g(index(i))+signs(i)*epsilon
        if(alpha(i)>0.and.alpha(i)<cap(i)) then
          fallback_sum=fallback_sum+value; fallback_count=fallback_count+1
        end if
        ! Roundoff near a bound must not turn a saturated solution's threshold
        ! interval into a spurious free-vector equality.
        bound_tol=32*spacing(1._dp)*cap(i)
        if(alpha(i)>bound_tol.and.alpha(i)<cap(i)-bound_tol) then
          free_sum=free_sum+value; count_free=count_free+1
        end if
        if((signs(i)>0.and.alpha(i)<cap(i)-bound_tol).or.(signs(i)<0.and.alpha(i)>bound_tol)) low=min(low,value)
        if((signs(i)>0.and.alpha(i)>bound_tol).or.(signs(i)<0.and.alpha(i)<cap(i)-bound_tol)) high=max(high,value)
      end do
      if(count_free>0) then
        levels(h)=free_sum/count_free
      else if((low==huge(1._dp).or.high==-huge(1._dp)).and.fallback_count>0) then
        levels(h)=fallback_sum/fallback_count
      else
        levels(h)=low/2+high/2
      end if
    end do
    self%dual_full_=beta; self%support_=pack([(i,i=1,n)],abs(beta)>0)
    self%n_support_=size(self%support_); self%dual_coef_=beta(self%support_)
    if(self%kernel_fit_=='precomputed') then
      self%support_vectors_=reshape([real(dp)::],[0,0])
    else if(self%sparse_fit_)then
      self%support_vectors_csr_=sparse_rows(self%x_fit_csr_,self%support_)
    else
      self%support_vectors_=self%x_fit_(self%support_,:)
    end if
    if(self%kernel_fit_=='linear')then
      if(self%sparse_fit_)then
        allocate(self%coef_(self%x_fit_csr_%n_cols));self%coef_=0
        do i=1,n
          do j=self%x_fit_csr_%indptr(i),self%x_fit_csr_%indptr(i+1)-1
            a=self%x_fit_csr_%indices(j)
            self%coef_(a)=self%coef_(a)+beta(i)*self%x_fit_csr_%data(j)
          end do
        end do
      else
        self%coef_=matmul(beta,self%x_fit_)
      end if
    end if
  end subroutine
  subroutine svr_fit(self,x,y,sample_weight)
    class(svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%epsilon),'invalid SVR epsilon')
    call require(self%epsilon>=0,'invalid SVR epsilon'); cap=bounds(size(y),self%c,sample_weight)
    call setup(self,x,gram); call optimize(self,gram,y,cap,1,self%epsilon,0._dp,levels)
    self%intercept_=-levels(1)
  end subroutine
  subroutine nusvr_fit(self,x,y,sample_weight)
    class(nu_svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%nu),'invalid NuSVR nu')
    call require(self%nu>0.and.self%nu<=1,'invalid NuSVR nu'); cap=bounds(size(y),self%c,sample_weight)
    call setup(self,x,gram); call optimize(self,gram,y,cap,2,0._dp,self%nu,levels)
    self%intercept_=-(levels(1)+levels(2))/2; self%epsilon_=max(0._dp,(levels(2)-levels(1))/2)
  end subroutine
  subroutine one_fit(self,x,sample_weight)
    class(one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call require(ieee_is_finite(self%nu),'invalid OneClassSVM nu')
    call require(self%nu>0.and.self%nu<1,'OneClassSVM requires 0 < nu < 1')
    call setup(self,x,gram); cap=bounds(size(x,1),1._dp,sample_weight)
    call optimize(self,gram,spread(0._dp,1,size(x,1)),cap,3,0._dp,self%nu,levels)
    self%offset_=levels(1); self%intercept_=-self%offset_
  end subroutine
  subroutine svr_fit_csr(self,x,y,sample_weight)
    class(svr),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::gram(:,:),cap(:)
    real(dp)::levels(2)
    call check_csr(x);call require(x%n_rows==size(y),'SVR CSR target count mismatch')
    call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%epsilon),'invalid SVR epsilon')
    call require(self%epsilon>=0,'invalid SVR epsilon');cap=bounds(size(y),self%c,sample_weight)
    call setup_csr(self,x,gram);call optimize(self,gram,y,cap,1,self%epsilon,0._dp,levels)
    call remap_sparse_support(self,sample_weight)
    self%intercept_=-levels(1)
  end subroutine
  subroutine nusvr_fit_csr(self,x,y,sample_weight)
    class(nu_svr),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::gram(:,:),cap(:)
    real(dp)::levels(2)
    call check_csr(x);call require(x%n_rows==size(y),'NuSVR CSR target count mismatch')
    call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%nu),'invalid NuSVR nu')
    call require(self%nu>0.and.self%nu<=1,'invalid NuSVR nu');cap=bounds(size(y),self%c,sample_weight)
    call setup_csr(self,x,gram);call optimize(self,gram,y,cap,2,0._dp,self%nu,levels)
    call remap_sparse_support(self,sample_weight)
    self%intercept_=-(levels(1)+levels(2))/2;self%epsilon_=max(0._dp,(levels(2)-levels(1))/2)
  end subroutine
  subroutine one_fit_csr(self,x,sample_weight)
    class(one_class_svm),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::gram(:,:),cap(:)
    real(dp)::levels(2)
    call require(ieee_is_finite(self%nu),'invalid OneClassSVM nu')
    call require(self%nu>0.and.self%nu<1,'OneClassSVM requires 0 < nu < 1')
    call setup_csr(self,x,gram);cap=bounds(x%n_rows,1._dp,sample_weight)
    call optimize(self,gram,spread(0._dp,1,x%n_rows),cap,3,0._dp,self%nu,levels)
    call remap_sparse_support(self,sample_weight)
    self%offset_=levels(1);self%intercept_=-self%offset_
  end subroutine
  subroutine remap_sparse_support(self,sample_weight)
    class(kernel_model),intent(inout)::self
    real(dp),optional,intent(in)::sample_weight(:)
    integer::i
    ! LIBSVM's sparse wrapper reports indices in its positive-weight problem.
    ! Support vector rows themselves still refer to the original design.
    if(present(sample_weight))then
      do i=1,size(self%support_)
        self%support_(i)=count(sample_weight(:self%support_(i))>0)
      end do
    end if
  end subroutine
  function kernel_raw_csr(self,x) result(values)
    class(kernel_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    type(csr_matrix)::query
    call require(allocated(self%dual_full_),'kernel SVM is not fitted')
    call require(self%sparse_fit_,'cannot use CSR input in kernel SVM trained on dense data')
    call check_csr(x)
    call require(x%n_cols==self%x_fit_csr_%n_cols,'kernel SVM CSR feature count mismatch')
    query=svm_sorted(x)
    values=matmul(sparse_gram(self,query,self%support_vectors_csr_,.false.),self%dual_coef_)
  end function
  function kernel_raw(self,x) result(values)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    call require(allocated(self%dual_full_),'kernel SVM is not fitted')
    if(self%sparse_fit_)then
      call check_x(x);values=kernel_raw_csr(self,csr_from_dense(x));return
    end if
    call check_xy(transpose(x),size(self%x_fit_,2))
    values=matmul(gram_matrix(self,x,self%x_fit_),self%dual_full_)
  end function
  function kernel_decision(self,x) result(values)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=kernel_raw(self,x)+self%intercept_
  end function
  function regression_predict(self,x) result(values)
    class(regression_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=self%decision_function(x)
  end function
  real(dp) function regression_score(self,x,y) result(value)
    class(regression_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
  function one_predict(self,x) result(labels)
    class(one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    labels=merge(1,-1,self%decision_function(x)>0)
  end function
  function one_score_samples(self,x) result(values)
    class(one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=kernel_raw(self,x)
  end function
  function one_fit_predict(self,x,sample_weight) result(labels)
    class(one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: labels(:)
    call self%fit(x,sample_weight); labels=self%predict(x)
  end function
  function kernel_decision_csr(self,x) result(values)
    class(kernel_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    values=kernel_raw_csr(self,x)+self%intercept_
  end function
  function regression_predict_csr(self,x) result(values)
    class(regression_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    values=self%decision_function(x)
  end function
  real(dp) function regression_score_csr(self,x,y) result(value)
    class(regression_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    value=r2_score(y,self%predict(x))
  end function
  function one_predict_csr(self,x) result(labels)
    class(one_class_svm),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::labels(:)
    labels=merge(1,-1,self%decision_function(x)>0)
  end function
  function one_score_samples_csr(self,x) result(values)
    class(one_class_svm),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    values=kernel_raw_csr(self,x)
  end function
  function one_fit_predict_csr(self,x,sample_weight) result(labels)
    class(one_class_svm),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    integer,allocatable::labels(:)
    call self%fit(x,sample_weight);labels=self%predict(x)
  end function
end module
