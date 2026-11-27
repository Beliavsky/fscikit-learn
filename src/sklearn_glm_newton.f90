! sklearn 1.6.1 _glm/_newton_solver.py, _linear_loss.py and _loss/_loss.pyx.tp
! provide BSD contracts/formulas. Native symmetric factorization/CSR glue is MIT.
module sklearn_glm_newton
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix
  use sklearn_glm_lbfgs,only:glm_loss_gradient,glm_lbfgs_solve,glm_raw_prediction,glm_suspend_ieee,glm_restore_ieee
  use sklearn_kernel_dispatch,only:kernel_blas_transpose_column,kernel_blas_inner_product
  use sklearn_boosting_loss,only:numpy_sum=>boosting_sum
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::glm_newton_solve,glm_gradient_hessian,glm_symmetric_step
contains
  subroutine glm_gradient_hessian(point,raw,y,weight,power,link,alpha,g,h,warning,dense,sparse)
    real(dp),intent(in)::point(:),raw(:),y(:),weight(:),power,alpha
    character(*),intent(in)::link
    real(dp),intent(out)::g(:),h(:,:)
    logical,intent(out)::warning
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::gp(size(y)),hp(size(y)),mu,a,b,mass
    integer::i,j,k,l,c,d,p,q
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    call glm_suspend_ieee(halting,raised)
    q=size(point);p=q
    if(present(dense))p=size(dense,2)
    if(present(sparse))p=sparse%n_cols
    do i=1,size(y)
      if(link=='log')then
        mu=exp(raw(i))
        if(power==0)then
          gp(i)=mu*(mu-y(i));hp(i)=mu*(2*mu-y(i))
        else if(power==1)then
          hp(i)=mu;gp(i)=mu-y(i)
        else if(power==2)then
          a=exp(-raw(i));gp(i)=1-y(i)*a;hp(i)=a*y(i)
        else
          a=exp((1-power)*raw(i));b=exp((2-power)*raw(i))
          gp(i)=b-y(i)*a;hp(i)=(2-power)*b-(1-power)*y(i)*a
        end if
      else
        mu=raw(i)
        if(power==0)then
          gp(i)=mu-y(i);hp(i)=1
        else if(power==1)then
          gp(i)=1-y(i)/mu;hp(i)=y(i)/(mu*mu)
        else if(power==2)then
          a=mu*mu;gp(i)=(mu-y(i))/a;hp(i)=(-1+2*y(i)/mu)/a
        else
          a=mu**(-power);gp(i)=a*(mu-y(i));hp(i)=a*((1-power)+power*y(i)/mu)
        end if
      end if
    end do
    mass=numpy_sum(weight);gp=gp*weight/mass;hp=hp*weight/mass
    warning=numpy_sum(merge(weight,0._dp,hp<=0))/mass>0.25_dp
    hp=abs(hp);g=0;h=0
    if(present(dense))then
      do j=1,p
        g(j)=kernel_blas_transpose_column(dense(:,j),gp,j,p)
      end do
    else
      do i=1,size(y)
        do k=sparse%indptr(i),sparse%indptr(i+1)-1
          j=sparse%indices(k);g(j)=g(j)+sparse%data(k)*gp(i)
        end do
      end do
    end if
    g(:p)=g(:p)+alpha*point(:p)
    if(q>p)g(q)=numpy_sum(gp)
    if(warning)then
      call glm_restore_ieee(halting,raised);return
    end if
    if(present(dense))then
      h(:p,:p)=matmul(transpose(dense),dense*spread(hp,2,p))
      if(q>p)then
        do j=1,p
          h(j,q)=kernel_blas_transpose_column(dense(:,j),hp,j,p)
        end do
        h(q,:p)=h(:p,q)
      end if
    else
      ! Pair stored entries within each row; duplicate cross terms are retained.
      do i=1,size(y)
        do k=sparse%indptr(i),sparse%indptr(i+1)-1
          c=sparse%indices(k)
          do l=sparse%indptr(i),sparse%indptr(i+1)-1
            d=sparse%indices(l);h(c,d)=h(c,d)+sparse%data(k)*hp(i)*sparse%data(l)
          end do
          if(q>p)h(c,q)=h(c,q)+sparse%data(k)*hp(i)
        end do
      end do
      if(q>p)h(q,:p)=h(:p,q)
    end if
    do j=1,p
      h(j,j)=h(j,j)+alpha
    end do
    if(q>p)h(q,q)=numpy_sum(hp)
    call glm_restore_ieee(halting,raised)
  end subroutine

  subroutine glm_symmetric_step(h,g,step,rcond,ok)
    ! The absolute pointwise curvature makes the GLM Hessian PSD. Use a native
    ! Cholesky solve and exact inverse one-norm condition check; Source uses
    ! LAPACK's symmetric factorization/condition estimate, so last bits may differ.
    real(dp),intent(in)::h(:,:),g(:)
    real(dp),intent(out)::step(:),rcond
    logical,intent(out)::ok
    real(dp)::lower(size(g),size(g)),rhs(size(g)),column(size(g)),a,norm_h,norm_inverse
    integer::n,i,j,k
    n=size(g);ok=.false.;rcond=0;step=0;lower=0
    if(.not.all(ieee_is_finite(h)).or..not.all(ieee_is_finite(g)))return
    do j=1,n
      a=h(j,j)-dot_product(lower(j,:j-1),lower(j,:j-1))
      if(a<=0.or..not.ieee_is_finite(a))return
      lower(j,j)=sqrt(a)
      do i=j+1,n
        lower(i,j)=(h(j,i)-dot_product(lower(i,:j-1),lower(j,:j-1)))/lower(j,j)
      end do
    end do
    norm_h=maxval(sum(abs(h),dim=1));norm_inverse=0
    do k=1,n
      rhs=0;rhs(k)=1;call triangular(rhs,column)
      norm_inverse=max(norm_inverse,sum(abs(column)))
    end do
    rcond=(1/norm_h)/norm_inverse
    if(.not.ieee_is_finite(rcond).or.rcond<0.5_dp*epsilon(1._dp))return
    call triangular(-g,step);ok=all(ieee_is_finite(step))
  contains
    subroutine triangular(b,x)
      real(dp),intent(in)::b(:)
      real(dp),intent(out)::x(:)
      real(dp)::z(size(b))
      integer::ii
      do ii=1,n
        z(ii)=(b(ii)-dot_product(lower(ii,:ii-1),z(:ii-1)))/lower(ii,ii)
      end do
      do ii=n,1,-1
        x(ii)=(z(ii)-dot_product(lower(ii+1:n,ii),x(ii+1:n)))/lower(ii,ii)
      end do
    end subroutine
  end subroutine

  subroutine glm_newton_solve(point,y,weight,power,link,alpha,tol,max_iter,n_iter,converged,message,dense,sparse)
    real(dp),intent(inout)::point(:)
    real(dp),intent(in)::y(:),weight(:),power,alpha,tol
    character(*),intent(in)::link
    integer,intent(in)::max_iter
    integer,intent(out)::n_iter
    logical,intent(out)::converged
    character(*),intent(out)::message
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::raw(size(y)),delta(size(y)),trial_raw(size(y)),loss,old_loss,rate,improvement,gd,rcond,decrement
    real(dp)::g(size(point)),old_g(size(point)),h(size(point),size(point)),step(size(point)),old(size(point))
    integer::iteration,ls,lb_iterations
    logical::warning,ok,accepted,halting(size(ieee_all)),raised(size(ieee_all))
    character(24)::reason
    character(60)::lb_message
    call glm_suspend_ieee(halting,raised)
    call glm_raw_prediction(point,raw,dense,sparse)
    call glm_loss_gradient(point,y,weight,power,link,alpha,loss,g,dense,sparse,raw)
    converged=.false.;reason='';iteration=1;message='Newton iteration limit'
    do while(iteration<=max_iter)
      call glm_gradient_hessian(point,raw,y,weight,power,link,alpha,g,h,warning,dense,sparse)
      if(warning)then
        reason='curvature';exit
      end if
      call glm_symmetric_step(h,g,step,rcond,ok)
      if(.not.ok)then
        reason='singular/ill-conditioned';exit
      end if
      gd=kernel_blas_inner_product(g,step)
      if(gd>0)then
        reason='non-descent';exit
      end if
      old=point;old_g=g;old_loss=loss;rate=1;accepted=.false.
      call glm_raw_prediction(step,delta,dense,sparse)
      do ls=0,20
        point=old+rate*step;trial_raw=raw+rate*delta
        call glm_loss_gradient(point,y,weight,power,link,alpha,loss,g,dense,sparse,trial_raw)
        improvement=loss-old_loss
        accepted=improvement<=rate*0.00048828125_dp*gd
        if(.not.accepted.and.abs(improvement)<=abs(old_loss*16*epsilon(1._dp))) &
          accepted=sum(abs(g))<sum(abs(old_g))
        if(accepted)exit
        rate=rate*0.5_dp
      end do
      if(.not.accepted)then
        reason='line-search';exit
      end if
      raw=trial_raw
      decrement=dot_product(step,matmul(h,step))
      converged=maxval(abs(g))<=tol.and.0.5_dp*decrement<=tol
      iteration=iteration+1
      if(converged)then
        message='Newton converged';exit
      end if
    end do
    if(len_trim(reason)>0)then
      call glm_lbfgs_solve(point,y,weight,power,link,alpha,tol,max_iter-iteration,lb_iterations, &
        converged,lb_message,dense,sparse)
      iteration=iteration+lb_iterations
      message='Newton->L-BFGS '//trim(reason)//': '//trim(lb_message)
    end if
    n_iter=iteration-1
    call glm_restore_ieee(halting,raised)
  end subroutine
end module
