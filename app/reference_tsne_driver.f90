program reference_tsne_driver
  use sklearn, only: tsne,trustworthiness,dp,pairwise_distances
  use sklearn_tsne_core, only: tsne_joint_probabilities,tsne_exact_objective,tsne_gradient_descent
  implicit none
  type(tsne) :: model
  real(dp), allocatable :: x(:,:),initial(:,:),distances(:,:),p(:),g(:,:),short(:,:),embedding(:,:)
  real(dp), allocatable :: trace(:,:),replay(:,:),replay_p(:)
  real(dp) :: error,short_error,rate,replay_error
  integer :: n,features,k,iterations,patience,seed,initialization,metric,unit,out,i,step,nshort,degree
  integer :: trace_unit,first,replay_iteration
  logical :: single
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) n,features,k,iterations,patience,seed,initialization,metric,single,nshort
  read(unit,*) model%perplexity,model%early_exaggeration,model%learning_rate,model%min_grad_norm
  allocate(x(n,features),initial(n,k))
  do i=1,n
    read(unit,*) x(i,:)
  end do
  do i=1,n
    read(unit,*) initial(i,:)
  end do
  close(unit)
  model%n_components=k; model%max_iter=iterations; model%n_iter_without_progress=patience; model%random_state=seed
  select case(metric)
  case(0); model%metric='euclidean'; distances=pairwise_distances(x,x,'sqeuclidean')
  case(1); model%metric='precomputed'; distances=x**2
  case(2); model%metric='manhattan'; distances=pairwise_distances(x,x,'manhattan')**2
  case(3); model%metric='cosine'; distances=pairwise_distances(x,x,'cosine')**2
  end select
  call tsne_joint_probabilities(distances,model%perplexity,p); degree=max(1,k-1)
  short=initial
  call tsne_exact_objective(short,p,degree,error,g,single)
  rate=model%learning_rate
  if(rate==-1) rate=max(real(n,dp)/model%early_exaggeration/4,50._dp)
  open(newunit=out,file=trim(output),status='replace')
  write(out,'(a,*(es26.17e3,1x))') 'probabilities ',p
  write(out,'(a,*(es26.17e3,1x))') 'objective ',error
  write(out,'(a,*(es26.17e3,1x))') 'gradient ',transpose(g)
  call tsne_gradient_descent(short,p*model%early_exaggeration,degree,0,nshort,patience,.5_dp,rate, &
    model%min_grad_norm,single,model%learning_rate==-1,short_error,step)
  write(out,'(a,*(es26.17e3,1x))') 'short_embedding ',transpose(short)
  write(out,'(a,*(es26.17e3,1x))') 'short_error ',short_error
  write(out,*) 'short_iteration ',step
  if(initialization==0) then
    embedding=model%fit_transform(x,initial)
  else
    model%init='random'
    if(initialization==2) model%init='pca'
    embedding=model%fit_transform(x)
  end if
  if(initialization==0) then
    replay=initial; replay_p=p*model%early_exaggeration
    open(newunit=trace_unit,file=trim(output)//'.trace',status='replace')
    call tsne_gradient_descent(replay,replay_p,degree,0,250,250,.5_dp,rate,model%min_grad_norm, &
      .false.,model%learning_rate==-1,replay_error,replay_iteration,trace)
    do i=1,size(trace,1)
      write(trace_unit,'(2(i0,1x),*(es26.17e3,1x))') 1,i-1,trace(i,:)
    end do
    replay_p=replay_p/model%early_exaggeration
    if(replay_iteration<250.or.model%max_iter>250) then
      first=replay_iteration+1
      call tsne_gradient_descent(replay,replay_p,degree,first,model%max_iter,model%n_iter_without_progress, &
        .8_dp,rate,model%min_grad_norm,.false.,model%learning_rate==-1,replay_error,replay_iteration,trace)
      do i=1,size(trace,1)
        write(trace_unit,'(2(i0,1x),*(es26.17e3,1x))') 2,first+i-1,trace(i,:)
      end do
    end if
    close(trace_unit)
    write(out,'(a,*(es26.17e3,1x))') 'replay_embedding ',transpose(replay)
    write(out,'(a,*(es26.17e3,1x))') 'replay_kl ',replay_error
    write(out,*) 'replay_iteration ',replay_iteration
  end if
  write(out,'(a,*(es26.17e3,1x))') 'embedding ',transpose(embedding)
  write(out,'(a,*(es26.17e3,1x))') 'kl ',model%kl_divergence_
  write(out,'(a,*(es26.17e3,1x))') 'learning_rate ',model%learning_rate_
  write(out,*) 'iteration ',model%n_iter_
  write(out,*) 'features ',model%n_features_in_
  write(out,'(a,*(es26.17e3,1x))') 'trust ', &
    trustworthiness(x,embedding,min(3,(n-1)/2),trim(model%metric))
  call tsne_exact_objective(embedding,p,degree,error,g)
  write(out,'(a,*(es26.17e3,1x))') 'final_objective ',error
  write(out,'(a,*(es26.17e3,1x))') 'final_gradient ',transpose(g)
  write(out,'(a,*(es26.17e3,1x))') 'preserved ',transpose(x),transpose(initial)
  close(out)
end program
