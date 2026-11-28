! LinearSVC/LinearSVR objectives: sklearn 1.6.1 svm/_classes.py and
! svm/src/liblinear/linear.cpp. BSD notices: LICENSES/scikit-learn-COPYING and liblinear-COPYRIGHT.
! Native LIBLINEAR-style TRON and seeded dual solvers; no liblinear binary is linked.
module sklearn_linear_svm
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index
  use sklearn_metrics, only: accuracy_score,r2_score
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_kernel_dispatch,only:tron_dot=>kernel_blas_inner_product
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::ieee_exceptions
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_fma
  implicit none
  private
  public :: linear_svc,linear_svr
  type :: linear_svc
    real(dp) :: c=1,tol=1.e-4_dp,intercept_scaling=1
    integer :: max_iter=1000,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,class_weight_balanced=.false.,converged_=.false.
    character(16)::loss='squared_hinge'
    character(8)::dual='auto'
    integer::random_state=0
    logical::store_dual_variables=.false.
    real(dp),allocatable::dual_variables_(:,:),dual_objective_(:),duality_gap_(:),kkt_violation_(:)
    integer, allocatable :: classes_(:),n_iter_per_class_(:)
    logical, allocatable :: converged_per_class_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:),objective_(:),gradient_norm_(:)
  contains
    procedure,private :: fit_dense=>linear_svc_fit,fit_csr=>linear_svc_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure,private :: decision_dense=>linear_svc_decision,decision_csr=>linear_svc_decision_csr
    generic :: decision_function=>decision_dense,decision_csr
    procedure,private :: predict_dense=>linear_svc_predict,predict_csr=>linear_svc_predict_csr
    generic :: predict=>predict_dense,predict_csr
    procedure,private :: score_dense=>linear_svc_score,score_csr=>linear_svc_score_csr
    generic :: score=>score_dense,score_csr
  end type
  type :: linear_svr
    real(dp) :: c=1,tol=1.e-4_dp,intercept_scaling=1,epsilon=0,intercept_=0,objective_=0,gradient_norm_=0
    integer :: max_iter=1000,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,converged_=.false.
    character(32)::loss='epsilon_insensitive'
    character(8)::dual='auto'
    integer::random_state=0
    real(dp)::kkt_violation_=0,dual_objective_=0,duality_gap_=0
    real(dp),allocatable::dual_variables_(:)
    real(dp), allocatable :: coef_(:)
  contains
    procedure,private :: fit_dense=>linear_svr_fit,fit_csr=>linear_svr_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure,private :: predict_dense=>linear_svr_predict,predict_csr=>linear_svr_predict_csr
    generic :: predict=>predict_dense,predict_csr
    procedure,private :: score_dense=>linear_svr_score,score_csr=>linear_svr_score_csr
    generic :: score=>score_dense,score_csr
  end type
contains
  logical function use_svc_dual(self,n,p)result(use_dual)
    class(linear_svc),intent(in)::self
    integer,intent(in)::n,p
    call require(self%loss=='hinge'.or.self%loss=='squared_hinge','invalid LinearSVC loss')
    call require(self%dual=='true'.or.self%dual=='false'.or.self%dual=='auto','invalid LinearSVC dual selector')
    call require(self%random_state>=0,'LinearSVC random_state must be nonnegative')
    use_dual=self%dual=='true'
    if(self%dual=='auto')use_dual=n<p.or.self%loss=='hinge'
    call require(use_dual.or.self%loss=='squared_hinge','LinearSVC hinge loss requires the dual solver')
  end function
  function svc_rows(x,rows)result(z)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::rows(:)
    type(csr_matrix)::z
    integer::i,a,b,k
    z%n_rows=size(rows);z%n_cols=x%n_cols;allocate(z%indptr(z%n_rows+1));z%indptr(1)=1
    do i=1,z%n_rows
      z%indptr(i+1)=z%indptr(i)+x%indptr(rows(i)+1)-x%indptr(rows(i))
    end do
    allocate(z%indices(z%indptr(z%n_rows+1)-1),z%data(z%indptr(z%n_rows+1)-1))
    do i=1,z%n_rows
      a=x%indptr(rows(i));b=x%indptr(rows(i)+1)-1;k=z%indptr(i)
      z%indices(k:k+b-a)=x%indices(a:b);z%data(k:k+b-a)=x%data(a:b)
    end do
  end function
  subroutine svc_dual_solve(x,y,cost,squared,intercept,scaling,tol,max_iter,rng,coef,alpha, &
      iterations,converged,objective,dual_objective,kkt)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),cost(:),scaling,tol
    logical,intent(in)::squared,intercept
    integer,intent(in)::max_iter
    type(legacy_random_state),intent(inout)::rng
    real(dp),allocatable,intent(out)::coef(:),alpha(:)
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp),intent(out)::objective,dual_objective,kkt
    real(dp),allocatable::qd(:),diagonal(:),upper(:),loss(:)
    integer,allocatable::order(:)
    real(dp)::g,pg,maximum,minimum,old_max,old_min,old_alpha,delta
    logical::modes(size(ieee_all)),flags(size(ieee_all)),shrink
    integer::n,d,i,j,k,s,active,it,swap
    n=x%n_rows;d=x%n_cols+merge(1,0,intercept)
    allocate(coef(d),alpha(n),qd(n),diagonal(n),upper(n),order(n));coef=0;alpha=0
    call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
    diagonal=0;upper=cost
    if(squared)then
      diagonal=.5_dp/cost;upper=huge(1._dp)
    end if
    do i=1,n
      qd(i)=diagonal(i)
      do k=x%indptr(i),x%indptr(i+1)-1
        qd(i)=qd(i)+x%data(k)*x%data(k)
      end do
      if(intercept)qd(i)=qd(i)+scaling*scaling
    end do
    order=[(i,i=1,n)];active=n;old_max=huge(1._dp);old_min=-huge(1._dp);iterations=0
    do it=1,max_iter
      maximum=-huge(1._dp);minimum=huge(1._dp)
      do i=1,active
        j=i+bounded_liblinear(rng,active-i+1);swap=order(i);order(i)=order(j);order(j)=swap
      end do
      s=1
      do while(s<=active)
        i=order(s);g=0
        do k=x%indptr(i),x%indptr(i+1)-1
          g=g+coef(x%indices(k))*x%data(k)
        end do
        if(intercept)g=g+coef(d)*scaling
        g=g*y(i)-1;g=g+alpha(i)*diagonal(i);pg=0;shrink=.false.
        if(alpha(i)==0)then
          if(g>old_max)then
            shrink=.true.
          else if(g<0)then
            pg=g
          end if
        else if(alpha(i)==upper(i))then
          if(g<old_min)then
            shrink=.true.
          else if(g>0)then
            pg=g
          end if
        else
          pg=g
        end if
        if(shrink)then
          swap=order(s);order(s)=order(active);order(active)=swap;active=active-1;cycle
        end if
        maximum=max(maximum,pg);minimum=min(minimum,pg)
        if(abs(pg)>1.e-12_dp)then
          old_alpha=alpha(i)
          if(qd(i)==0)then
            if(g<0)then
              alpha(i)=upper(i)
            else
              alpha(i)=0
            end if
          else
            alpha(i)=min(max(alpha(i)-g/qd(i),0._dp),upper(i))
          end if
          delta=(alpha(i)-old_alpha)*y(i)
          do k=x%indptr(i),x%indptr(i+1)-1
            j=x%indices(k);coef(j)=coef(j)+delta*x%data(k)
          end do
          if(intercept)coef(d)=coef(d)+delta*scaling
        end if
        s=s+1
      end do
      iterations=it
      if(maximum-minimum<=tol)then
        if(active==n)exit
        active=n;old_max=huge(1._dp);old_min=-huge(1._dp);cycle
      end if
      old_max=maximum;old_min=minimum
      if(old_max<=0)old_max=huge(1._dp)
      if(old_min>=0)old_min=-huge(1._dp)
    end do
    converged=iterations<max_iter
    loss=max(0._dp,1-y*sparse_forward(x,coef,intercept,scaling))
    objective=.5_dp*sum(coef**2)
    if(squared)then
      objective=objective+sum(cost*loss**2)
    else
      objective=objective+sum(cost*loss)
    end if
    dual_objective=sum(alpha)-.5_dp*sum(coef**2)-.5_dp*sum(diagonal*alpha**2);kkt=0
    do i=1,n
      g=0
      do k=x%indptr(i),x%indptr(i+1)-1
        g=g+coef(x%indices(k))*x%data(k)
      end do
      if(intercept)g=g+coef(d)*scaling
      g=g*y(i)-1+alpha(i)*diagonal(i)
      if(alpha(i)==0)g=min(g,0._dp)
      if(alpha(i)==upper(i))g=max(g,0._dp)
      kkt=max(kkt,abs(g))
    end do
    call ieee_set_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_all,flags)
  end subroutine
  subroutine svr_tron_state(x,y,cost,coef,tube,intercept,scaling,objective,gradient,active,classification)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),cost(:),coef(:),tube,scaling
    logical,intent(in)::intercept
    real(dp),intent(out)::objective,gradient(:)
    logical,intent(out)::active(:)
    logical,optional,intent(in)::classification
    real(dp)::margin,residual,value
    integer::i,j,k,d
    d=size(coef);objective=0;gradient=0;active=.false.
    do j=1,d
      objective=objective+coef(j)*coef(j)
    end do
    objective=objective/2
    do i=1,x%n_rows
      if(cost(i)==0)cycle
      margin=0
      do k=x%indptr(i),x%indptr(i+1)-1
        margin=margin+coef(x%indices(k))*x%data(k)
      end do
      if(intercept)margin=margin+coef(d)*scaling
      if(present(classification))then
        if(classification)then
          residual=y(i)*margin
          if(residual>=1)cycle
          active(i)=.true.;value=cost(i)*y(i)*(residual-1)
          objective=objective+(cost(i)*(1-residual))*(1-residual)
        else
          call regression_loss()
        end if
      else
        call regression_loss()
      end if
      if(.not.active(i))cycle
      do k=x%indptr(i),x%indptr(i+1)-1
        j=x%indices(k);gradient(j)=gradient(j)+value*x%data(k)
      end do
      if(intercept)gradient(d)=gradient(d)+value*scaling
    end do
    gradient=coef+2*gradient
  contains
    subroutine regression_loss()
      residual=margin-y(i)
      if(residual<-tube)then
        residual=residual+tube
      else if(residual>tube)then
        residual=residual-tube
      else
        return
      end if
      active(i)=.true.;value=cost(i)*residual;objective=objective+value*residual
    end subroutine
  end subroutine
  subroutine svr_tron_hv(x,cost,active,vector,intercept,scaling,product)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::cost(:),vector(:),scaling
    logical,intent(in)::active(:),intercept
    real(dp),intent(out)::product(:)
    real(dp)::value
    integer::i,j,k,d
    d=size(vector);product=0
    do i=1,x%n_rows
      if(.not.active(i))cycle
      value=0
      do k=x%indptr(i),x%indptr(i+1)-1
        value=value+vector(x%indices(k))*x%data(k)
      end do
      if(intercept)value=value+vector(d)*scaling
      value=cost(i)*value
      do k=x%indptr(i),x%indptr(i+1)-1
        j=x%indices(k);product(j)=product(j)+value*x%data(k)
      end do
      if(intercept)product(d)=product(d)+value*scaling
    end do
    product=vector+2*product
  end subroutine
  subroutine svr_tron_cg(x,cost,active,gradient,intercept,scaling,delta,step,residual)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::cost(:),gradient(:),scaling,delta
    logical,intent(in)::active(:),intercept
    real(dp),intent(out)::step(:),residual(:)
    real(dp)::direction(size(gradient)),hv(size(gradient)),rtr,next,alpha,beta,tolerance,sd,ss,dd,rad
    integer::j
    step=0;residual=-gradient;direction=residual
    tolerance=.1_dp*norm2(gradient);rtr=tron_dot(residual,residual)
    do
      if(norm2(residual)<=tolerance)exit
      call svr_tron_hv(x,cost,active,direction,intercept,scaling,hv)
      alpha=rtr/tron_dot(direction,hv)
      do j=1,size(step)
        step(j)=ieee_fma(alpha,direction(j),step(j))
      end do
      if(norm2(step)>delta)then
        do j=1,size(step)
          step(j)=ieee_fma(-alpha,direction(j),step(j))
        end do
        sd=tron_dot(step,direction);ss=tron_dot(step,step);dd=tron_dot(direction,direction)
        rad=sqrt(sd*sd+dd*(delta*delta-ss))
        if(sd>=0)then
          alpha=(delta*delta-ss)/(sd+rad)
        else
          alpha=(rad-sd)/dd
        end if
        do j=1,size(step)
          step(j)=ieee_fma(alpha,direction(j),step(j))
          residual(j)=ieee_fma(-alpha,hv(j),residual(j))
        end do
        exit
      end if
      do j=1,size(step)
        residual(j)=ieee_fma(-alpha,hv(j),residual(j))
      end do
      next=tron_dot(residual,residual);beta=next/rtr
      direction=beta*direction+residual;rtr=next
    end do
  end subroutine
  subroutine svr_tron_solve(x,y,cost,tube,intercept,scaling,tol,max_iter,coef,iterations,converged,objective,gradnorm, &
      classification)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),cost(:),tube,scaling,tol
    logical,intent(in)::intercept
    integer,intent(in)::max_iter
    real(dp),allocatable,intent(out)::coef(:)
    integer,intent(out)::iterations
    logical,intent(out)::converged
    logical,optional,intent(in)::classification
    real(dp),intent(out)::objective,gradnorm
    real(dp),allocatable::gradient(:),trial_g(:),step(:),residual(:),candidate(:)
    logical,allocatable::active(:),trial_active(:)
    real(dp)::delta,initial_norm,trial,gs,predicted,actual,snorm,alpha
    integer::d,it
    call require(x%n_cols<huge(1).or..not.intercept,'LinearSVR synthetic feature exceeds integer range')
    d=x%n_cols+merge(1,0,intercept)
    allocate(coef(d),gradient(d),trial_g(d),step(d),residual(d),candidate(d),active(x%n_rows),trial_active(x%n_rows))
    coef=0;call svr_tron_state(x,y,cost,coef,tube,intercept,scaling,objective,gradient,active,classification)
    delta=norm2(gradient);initial_norm=delta;gradnorm=delta;it=1
    do while(it<=max_iter.and.gradnorm>tol*initial_norm)
      call svr_tron_cg(x,cost,active,gradient,intercept,scaling,delta,step,residual)
      candidate=coef+step;gs=tron_dot(gradient,step)
      predicted=-.5_dp*(gs-tron_dot(step,residual))
      call svr_tron_state(x,y,cost,candidate,tube,intercept,scaling,trial,trial_g,trial_active,classification)
      actual=objective-trial;snorm=norm2(step)
      if(it==1)delta=min(delta,snorm)
      if(trial-objective-gs<=0)then
        alpha=4
      else
        alpha=max(.25_dp,-.5_dp*(gs/(trial-objective-gs)))
      end if
      if(actual<1.e-4_dp*predicted)then
        delta=min(max(alpha,.25_dp)*snorm,.5_dp*delta)
      else if(actual<.25_dp*predicted)then
        delta=max(.25_dp*delta,min(alpha*snorm,.5_dp*delta))
      else if(actual<.75_dp*predicted)then
        delta=max(.25_dp*delta,min(alpha*snorm,4*delta))
      else
        delta=max(delta,min(alpha*snorm,4*delta))
      end if
      if(actual>1.e-4_dp*predicted)then
        it=it+1;coef=candidate;objective=trial;gradient=trial_g;active=trial_active;gradnorm=norm2(gradient)
        if(gradnorm<=tol*initial_norm)exit
      end if
      if(objective<-1.e32_dp)exit
      if(abs(actual)<=0.and.predicted<=0)exit
      if(abs(actual)<=1.e-12_dp*abs(objective).and.abs(predicted)<=1.e-12_dp*abs(objective))exit
    end do
    iterations=it-1;converged=iterations<max_iter
  end subroutine
  logical function use_svr_dual(self,n,p)result(use_dual)
    class(linear_svr),intent(in)::self
    integer,intent(in)::n,p
    call require(self%loss=='epsilon_insensitive'.or.self%loss=='squared_epsilon_insensitive','invalid LinearSVR loss')
    call require(self%dual=='true'.or.self%dual=='false'.or.self%dual=='auto','invalid LinearSVR dual selector')
    call require(self%random_state>=0,'LinearSVR random_state must be nonnegative')
    use_dual=self%dual=='true'
    if(self%dual=='auto')use_dual=n<p.or.self%loss=='epsilon_insensitive'
    call require(use_dual.or.self%loss=='squared_epsilon_insensitive', &
      'LinearSVR epsilon_insensitive loss requires the dual solver')
  end function
  integer function bounded_liblinear(rng,n)result(value)
    type(legacy_random_state),intent(inout)::rng
    integer,intent(in)::n
    integer(int64)::product,low,threshold
    ! newrand.h: MT19937 with Lemire rejection, including a draw for n=1.
    threshold=modulo(4294967296_int64,int(n,int64))
    do
      product=rng%raw_uint32()*int(n,int64);low=iand(product,int(z'FFFFFFFF',int64))
      if(low>=threshold)exit
    end do
    value=int(ishft(product,-32))
  end function
  real(dp) function coordinate_violation(g,beta,upper,tube)result(violation)
    real(dp),intent(in)::g,beta,upper,tube
    violation=0
    if(beta==0)then
      if(g+tube<0)then
        violation=-g-tube
      else if(g-tube>0)then
        violation=g-tube
      end if
    else if(beta>=upper)then
      if(g+tube>0)violation=g+tube
    else if(beta<=-upper)then
      if(g-tube<0)violation=-g+tube
    else if(beta>0)then
      violation=abs(g+tube)
    else
      violation=abs(g-tube)
    end if
  end function
  subroutine svr_dual_solve(self,x,y,w,coef)
    class(linear_svr),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),w(:)
    real(dp),allocatable,intent(out)::coef(:)
    type(legacy_random_state)::rng,seed_rng
    integer,allocatable::rows(:),order(:)
    real(dp),allocatable::beta(:),qd(:),lambda(:),upper(:),cost(:),residual(:)
    real(dp)::g,h,gp,gn,violation,maximum,old_max,norm,initial_norm,step,old_beta,dual_penalty
    logical::modes(size(ieee_all)),flags(size(ieee_all)),squared,shrink
    integer::n,d,i,j,k,s,row,active,it,swap
    call require(x%n_cols<huge(1).or..not.self%fit_intercept,'LinearSVR synthetic feature exceeds integer range')
    call seed_rng%seed(self%random_state);call rng%seed(seed_rng%integer_index(huge(1))-1)
    rows=pack([(i,i=1,x%n_rows)],w>0);n=size(rows);d=x%n_cols+merge(1,0,self%fit_intercept)
    allocate(coef(d),beta(n),qd(n),lambda(n),upper(n),cost(n),order(n));coef=0;beta=0
    squared=self%loss=='squared_epsilon_insensitive'
    call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
    do i=1,n
      row=rows(i);cost(i)=self%c*w(row);qd(i)=0
      do k=x%indptr(row),x%indptr(row+1)-1
        qd(i)=qd(i)+x%data(k)**2
      end do
      if(self%fit_intercept)qd(i)=qd(i)+self%intercept_scaling**2
    end do
    lambda=0;upper=cost
    if(squared)then
      lambda=.5_dp/cost;upper=huge(1._dp)
    end if
    order=[(i,i=1,n)];active=n;old_max=huge(1._dp);initial_norm=-1;self%n_iter_=0
    do it=1,self%max_iter
      maximum=0;norm=0
      do i=1,active
        j=i+bounded_liblinear(rng,active-i+1);swap=order(i);order(i)=order(j);order(j)=swap
      end do
      s=1
      do while(s<=active)
        i=order(s);row=rows(i);g=-y(row)+lambda(i)*beta(i);h=qd(i)+lambda(i)
        do k=x%indptr(row),x%indptr(row+1)-1
          g=g+x%data(k)*coef(x%indices(k))
        end do
        if(self%fit_intercept)g=g+self%intercept_scaling*coef(d)
        gp=g+self%epsilon;gn=g-self%epsilon;shrink=.false.
        if(beta(i)==0)then
          shrink=gp>old_max.and.gn<-old_max
        else if(beta(i)>=upper(i))then
          shrink=gp<-old_max
        else if(beta(i)<=-upper(i))then
          shrink=gn>old_max
        end if
        if(shrink)then
          swap=order(s);order(s)=order(active);order(active)=swap;active=active-1;cycle
        end if
        violation=coordinate_violation(g,beta(i),upper(i),self%epsilon)
        maximum=max(maximum,violation);norm=norm+violation
        if(gp<h*beta(i))then
          if(h==0)then
            step=upper(i)-beta(i)
          else
            step=-gp/h
          end if
        else if(gn>h*beta(i))then
          if(h==0)then
            step=-upper(i)-beta(i)
          else
            step=-gn/h
          end if
        else
          step=-beta(i)
        end if
        if(.not.(abs(step)<1.e-12_dp))then
          old_beta=beta(i);beta(i)=min(max(beta(i)+step,-upper(i)),upper(i));step=beta(i)-old_beta
          if(step/=0)then
            do k=x%indptr(row),x%indptr(row+1)-1
              j=x%indices(k);coef(j)=coef(j)+step*x%data(k)
            end do
            if(self%fit_intercept)coef(d)=coef(d)+step*self%intercept_scaling
          end if
        end if
        s=s+1
      end do
      if(it==1)initial_norm=norm
      self%n_iter_=it
      if(norm<=self%tol*initial_norm)then
        if(active==n)exit
        active=n;old_max=huge(1._dp);cycle
      end if
      old_max=maximum
    end do
    self%converged_=self%n_iter_<self%max_iter
    self%dual_variables_=spread(0._dp,1,x%n_rows);self%dual_variables_(rows)=beta
    residual=max(0._dp,abs(sparse_forward(x,coef,self%fit_intercept,self%intercept_scaling)-y)-self%epsilon)
    self%objective_=.5_dp*sum(coef**2)
    if(squared)then
      self%objective_=self%objective_+self%c*sum(w*residual**2)
    else
      self%objective_=self%objective_+self%c*sum(w*residual)
    end if
    dual_penalty=0
    do i=1,n
      if(beta(i)/=0)dual_penalty=dual_penalty+.5_dp*lambda(i)*beta(i)**2
    end do
    self%dual_objective_=dot_product(y(rows),beta)-self%epsilon*sum(abs(beta))-.5_dp*sum(coef**2)-dual_penalty
    self%duality_gap_=self%objective_-self%dual_objective_;self%kkt_violation_=0
    do i=1,n
      row=rows(i);g=-y(row)+lambda(i)*beta(i)
      do k=x%indptr(row),x%indptr(row+1)-1
        g=g+x%data(k)*coef(x%indices(k))
      end do
      if(self%fit_intercept)g=g+self%intercept_scaling*coef(d)
      self%kkt_violation_=max(self%kkt_violation_,coordinate_violation(g,beta(i),upper(i),self%epsilon))
    end do
    self%gradient_norm_=self%kkt_violation_
    call ieee_set_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_all,flags)
  end subroutine
  subroutine clear_svr_dual(self)
    class(linear_svr),intent(inout)::self
    if(allocated(self%dual_variables_))deallocate(self%dual_variables_)
    self%dual_objective_=0;self%duality_gap_=0;self%kkt_violation_=0
  end subroutine
  subroutine check_sparse(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'linear SVM CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'linear SVM CSR input must be finite')
  end subroutine
  subroutine sparse_setup(x,c,tol,scaling,max_iter,w,sample_weight)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::c,tol,scaling
    integer,intent(in)::max_iter
    real(dp),allocatable,intent(out)::w(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call check_sparse(x)
    call require(ieee_is_finite(c),'nonfinite linear SVM C')
    call require(c>0,'linear SVM C must be positive')
    call require(ieee_is_finite(tol),'nonfinite linear SVM tolerance')
    call require(tol>0,'linear SVM tolerance must be positive')
    call require(ieee_is_finite(scaling),'nonfinite linear SVM intercept scaling')
    call require(scaling>0,'linear SVM intercept scaling must be positive')
    call require(max_iter>0,'linear SVM max_iter must be positive')
    allocate(w(x%n_rows));w=1
    if(present(sample_weight))then
      call require(size(sample_weight)==x%n_rows,'linear SVM sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite linear SVM sample weights')
      call require(all(sample_weight>=0),'negative linear SVM sample weights');w=sample_weight
    end if
    call require(sum(w)>0,'linear SVM needs positive total weight')
  end subroutine
  function sparse_forward(x,coef,intercept,scaling) result(values)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::coef(:),scaling
    logical,intent(in)::intercept
    real(dp)::values(x%n_rows)
    integer::i,k
    values=0
    do i=1,x%n_rows
      do k=x%indptr(i),x%indptr(i+1)-1
        values(i)=values(i)+x%data(k)*coef(x%indices(k))
      end do
    end do
    if(intercept)values=values+scaling*coef(x%n_cols+1)
  end function
  function sparse_transpose(x,values,intercept,scaling) result(coef)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::values(:),scaling
    logical,intent(in)::intercept
    real(dp)::coef(x%n_cols+merge(1,0,intercept))
    integer::i,j,k
    coef=0
    do i=1,x%n_rows
      do k=x%indptr(i),x%indptr(i+1)-1
        j=x%indices(k);coef(j)=coef(j)+values(i)*x%data(k)
      end do
    end do
    if(intercept)coef(x%n_cols+1)=scaling*sum(values)
  end function
  subroutine linear_svc_fit(self,x,y,sample_weight,class_weight)
    class(linear_svc), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    call check_xy(x,size(y))
    call linear_svc_fit_csr(self,csr_from_dense(x),y,sample_weight,class_weight)
  end subroutine

  function linear_svc_decision(self,x) result(scores)
    class(linear_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    call require(allocated(self%coef_),'LinearSVC is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'LinearSVC feature count mismatch')
    scores=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function

  function linear_svc_predict(self,x) result(y)
    class(linear_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: scores(:,:)
    scores=self%decision_function(x)
    if(size(scores,2)==1) then
      y=self%classes_(merge(2,1,scores(:,1)>0))
    else
      y=self%classes_(maxloc(scores,dim=2))
    end if
  end function

  real(dp) function linear_svc_score(self,x,y) result(value)
    class(linear_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function

  subroutine linear_svr_fit(self,x,y,sample_weight)
    class(linear_svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: w(:),coef(:)
    type(csr_matrix)::sparse
    integer :: p
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1])); p=size(x,2)
    call require(ieee_is_finite(self%epsilon),'nonfinite LinearSVR epsilon')
    call require(self%epsilon>=0,'LinearSVR epsilon must be nonnegative')
    if(use_svr_dual(self,size(y),p))then
      sparse=csr_from_dense(x)
      call sparse_setup(sparse,self%c,self%tol,self%intercept_scaling,self%max_iter,w,sample_weight)
      call svr_dual_solve(self,sparse,y,w,coef)
    else
      call clear_svr_dual(self)
      sparse=csr_from_dense(x)
      call sparse_setup(sparse,self%c,self%tol,self%intercept_scaling,self%max_iter,w,sample_weight)
      call svr_tron_solve(sparse,y,self%c*w,self%epsilon,self%fit_intercept,self%intercept_scaling,self%tol, &
        self%max_iter,coef,self%n_iter_,self%converged_,self%objective_,self%gradient_norm_)
    end if
    self%coef_=coef(:p); self%intercept_=0
    if(self%fit_intercept) self%intercept_=coef(p+1)*self%intercept_scaling
    self%n_features_in_=p
  end subroutine

  function linear_svr_predict(self,x) result(y)
    class(linear_svr), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call require(allocated(self%coef_),'LinearSVR is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'LinearSVR feature count mismatch')
    y=matmul(x,self%coef_)+self%intercept_
  end function

  real(dp) function linear_svr_score(self,x,y) result(value)
    class(linear_svr), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
  subroutine linear_svc_fit_csr(self,x,y,sample_weight,class_weight)
    class(linear_svc),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),allocatable::w(:),cw(:),target(:),cost(:),coef(:),alpha(:)
    integer,allocatable::encoded(:),rows(:),selected(:),active_classes(:)
    type(csr_matrix)::grouped
    type(legacy_random_state)::rng,seed_rng
    logical::dual
    real(dp)::primal_tol
    integer::n,p,c,k,i,channels,offset,ac,positive
    call check_sparse(x);call require(x%n_rows==size(y),'LinearSVC CSR target count mismatch')
    n=size(y);p=x%n_cols
    call sparse_setup(x,self%c,self%tol,self%intercept_scaling,self%max_iter,w,sample_weight)
    dual=use_svc_dual(self,n,p)
    call require(p<huge(1).or..not.self%fit_intercept,'LinearSVC synthetic feature exceeds integer range')
    self%classes_=unique_labels(y);c=size(self%classes_)
    call require(c>=2,'LinearSVC requires at least two classes')
    encoded=[(label_index(self%classes_,y(i)),i=1,n)];cw=spread(1._dp,1,c)
    if(present(class_weight))then
      call require(.not.self%class_weight_balanced,'choose balanced or explicit LinearSVC class weights')
      call require(size(class_weight)==c,'LinearSVC class weight count mismatch')
      call require(all(ieee_is_finite(class_weight)),'nonfinite LinearSVC class weights')
      call require(all(class_weight>0),'LinearSVC class weights must be positive');cw=class_weight
    else if(self%class_weight_balanced)then
      do k=1,c
        cw(k)=real(n,dp)/(c*real(count(encoded==k),dp))
      end do
    end if
    active_classes=pack([(i,i=1,c)],[(any((w>0).and.(encoded==i)),i=1,c)]);ac=size(active_classes)
    allocate(rows(count(w>0)));offset=0
    do k=1,ac
      selected=pack([(i,i=1,n)],(w>0).and.(encoded==active_classes(k)))
      rows(offset+1:offset+size(selected))=selected;offset=offset+size(selected)
    end do
    grouped=svc_rows(x,rows);channels=ac;if(ac<=2)channels=1
    call seed_rng%seed(self%random_state);call rng%seed(seed_rng%integer_index(huge(1))-1)
    if(allocated(self%coef_))deallocate(self%coef_,self%intercept_,self%objective_,self%gradient_norm_, &
      self%n_iter_per_class_,self%converged_per_class_)
    allocate(self%coef_(channels,p),self%intercept_(channels),self%objective_(channels),self%gradient_norm_(channels))
    allocate(self%n_iter_per_class_(channels),self%converged_per_class_(channels))
    if(allocated(self%dual_variables_))deallocate(self%dual_variables_)
    if(allocated(self%dual_objective_))deallocate(self%dual_objective_,self%duality_gap_,self%kkt_violation_)
    allocate(self%dual_objective_(channels),self%duality_gap_(channels),self%kkt_violation_(channels))
    self%dual_objective_=0;self%duality_gap_=0;self%kkt_violation_=0
    if(dual.and.self%store_dual_variables)then
      allocate(self%dual_variables_(n,channels));self%dual_variables_=0
    end if
    do k=1,channels
      if(ac==2)then
        target=merge(1._dp,-1._dp,encoded(rows)==active_classes(2));cost=w(rows)*(self%c*cw(encoded(rows)))
      else
        target=merge(1._dp,-1._dp,encoded(rows)==active_classes(k))
        cost=w(rows)*(self%c*merge(cw(active_classes(k)),1._dp,encoded(rows)==active_classes(k)))
      end if
      if(dual)then
        call svc_dual_solve(grouped,target,cost,self%loss=='squared_hinge',self%fit_intercept,self%intercept_scaling, &
          self%tol,self%max_iter,rng,coef,alpha,self%n_iter_per_class_(k),self%converged_per_class_(k), &
          self%objective_(k),self%dual_objective_(k),self%kkt_violation_(k))
        self%duality_gap_(k)=self%objective_(k)-self%dual_objective_(k);self%gradient_norm_(k)=self%kkt_violation_(k)
        if(self%store_dual_variables)self%dual_variables_(rows,k)=alpha
      else
        positive=count(target>0)
        primal_tol=self%tol*real(max(min(positive,size(target)-positive),1),dp)/size(target)
        call svr_tron_solve(grouped,target,cost,0._dp,self%fit_intercept,self%intercept_scaling,primal_tol, &
          self%max_iter,coef,self%n_iter_per_class_(k),self%converged_per_class_(k),self%objective_(k), &
          self%gradient_norm_(k),classification=.true.)
      end if
      self%coef_(k,:)=coef(:p);self%intercept_(k)=0
      if(self%fit_intercept)self%intercept_(k)=coef(p+1)*self%intercept_scaling
    end do
    self%n_features_in_=p;self%n_iter_=maxval(self%n_iter_per_class_);self%converged_=all(self%converged_per_class_)
  end subroutine
  subroutine linear_svr_fit_csr(self,x,y,sample_weight)
    class(linear_svr),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::w(:),coef(:)
    call check_sparse(x);call require(x%n_rows==size(y),'LinearSVR CSR target count mismatch')
    call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%epsilon),'nonfinite LinearSVR epsilon')
    call require(self%epsilon>=0,'LinearSVR epsilon must be nonnegative')
    call sparse_setup(x,self%c,self%tol,self%intercept_scaling,self%max_iter,w,sample_weight)
    if(use_svr_dual(self,size(y),x%n_cols))then
      call svr_dual_solve(self,x,y,w,coef)
    else
      call clear_svr_dual(self)
      call svr_tron_solve(x,y,self%c*w,self%epsilon,self%fit_intercept,self%intercept_scaling,self%tol, &
        self%max_iter,coef,self%n_iter_,self%converged_,self%objective_,self%gradient_norm_)
    end if
    self%coef_=coef(:x%n_cols);self%intercept_=0
    if(self%fit_intercept)self%intercept_=coef(x%n_cols+1)*self%intercept_scaling
    self%n_features_in_=x%n_cols
  end subroutine
  function linear_svc_decision_csr(self,x)result(scores)
    class(linear_svc),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::scores(:,:)
    call require(allocated(self%coef_),'LinearSVC is not fitted');call check_sparse(x)
    call require(x%n_cols==self%n_features_in_,'LinearSVC feature count mismatch')
    scores=x%matmul(transpose(self%coef_))+spread(self%intercept_,1,x%n_rows)
  end function
  function linear_svc_predict_csr(self,x)result(y)
    class(linear_svc),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::y(:)
    real(dp),allocatable::scores(:,:)
    scores=self%decision_function(x)
    if(size(scores,2)==1)then
      y=self%classes_(merge(2,1,scores(:,1)>0))
    else
      y=self%classes_(maxloc(scores,dim=2))
    end if
  end function
  real(dp) function linear_svc_score_csr(self,x,y)result(value)
    class(linear_svc),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    value=accuracy_score(y,self%predict(x))
  end function
  function linear_svr_predict_csr(self,x)result(y)
    class(linear_svr),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:),values(:,:)
    call require(allocated(self%coef_),'LinearSVR is not fitted');call check_sparse(x)
    call require(x%n_cols==self%n_features_in_,'LinearSVR feature count mismatch')
    values=x%matmul(reshape(self%coef_,[x%n_cols,1]));y=values(:,1)+self%intercept_
  end function
  real(dp) function linear_svr_score_csr(self,x,y)result(value)
    class(linear_svr),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    value=r2_score(y,self%predict(x))
  end function
end module
